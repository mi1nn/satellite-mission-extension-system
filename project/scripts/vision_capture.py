#!/usr/bin/env python3
"""MRV: AprilTag vision capture of a free-floating 3 t MEP (linear drift or 6-DoF).

Must run with the Isaac Sim Python (the same interpreter `srb` uses):

    cd ~/space_robotics_bench
    # Test 1 -- static pose accuracy (MEP at rest)
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario static --headless
    # Test 2 + 3 -- linear drift intercept, capture, 10 s holding, slow retreat
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless
    # 6-DoF -- XYZ drift + combined roll/pitch/yaw rate (mep.angular_velocity_rad_s)
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless --tag six_dof --set mep.motion_mode=six_dof
    # Full pipeline (the default: 6-DoF MEP, start yaw 15 deg, MRV rendezvous + capture + moving-client
    # docking, ROS 2 on, tag full_6dof) -- --no_dock / --no_moving_dock / --no_ros / --start_yaw_deg 0 /
    # --tag NAME switch those off or change them
    ~/isaac-sim/python.sh project/scripts/vision_capture.py
    # Moving client (on by default) -- the satellite drifts (+X 0.02 m/s), MRV + MEP match its velocity,
    # dock while moving, release the MEP and depart (-X); --dock_only skips the capture
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless --tag moving --dock_only
    # ... docking to a satellite at rest instead (docked hold `docking.hold_duration_s`, then SUCCESS)
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --no_moving_dock
    # ... the same run without the rendezvous phase (the arm starts at the observation pose)
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --dock --no_mrv_approach
    # Astrobee observation camera (on by default): flies around the satellite, image on
    # /astrobee/camera/image_raw (with ROS 2); --no_astrobee leaves it out
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --no_astrobee
    # ... it also maps the satellite from depth and checks the docking corridor; debris
    # floating at the docking port must end the run in DOCKING_UNAVAILABLE (not --dock_only:
    # the MEP then hides the port from the Astrobee from the start)
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --no_moving_dock --nozzle_obstruction --tag blocked
    # GUI (debug draw + camera overlay images); after a success (or DOCKING_UNAVAILABLE) the simulation keeps
    # running (robot holding, Astrobee scanning) until the window is closed (--exit_when_done to quit)
    ~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic

Config: `project/config/vision_capture.yaml`; override any value with
`--set section.key=value` (e.g. `--set mep.linear_velocity_mps=0.02`).
Outputs: `project/logs/vision_capture/<tag>_{result.json,metrics.csv,overlay/}` (tag = --tag or the scenario).
Exit code 0 only if every check of the scenario passed.
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def parse_args():
    isaac_path = os.environ.get("ISAAC_PATH", os.path.expanduser("~/isaac-sim"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=("static", "dynamic"), default="dynamic")
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim window")
    parser.add_argument("--config", type=str, default=None, help="Vision config YAML (default: project/config/vision_capture.yaml)")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE", help="Override a config value (repeatable)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: project/logs/vision_capture)")
    parser.add_argument("--tag", type=str, default=None, help="Run name for the output files (default: the scenario name)")
    parser.add_argument("--exit_when_done", action="store_true",
                        help="GUI: close the app when the scenario ends (default: after a success the simulation keeps running until the window is closed)")
    parser.add_argument("--dock", action="store_true",
                        help="Run the Ares1 probe -> satellite thruster docking phase after the capture (default: on, config `docking:`)")
    parser.add_argument("--no_dock", action="store_true", help="Capture only: skip the docking phase")
    parser.add_argument("--dock_only", action="store_true",
                        help="--dock, but skip the capture: attach the MEP at its nominal grasp pose and dock straight away")
    parser.add_argument("--moving_dock", action="store_true",
                        help="Docking with a drifting client (the default whenever docking runs): release -> chase -> "
                             "velocity matching -> rendezvous -> docking -> robot release -> MRV departure (config "
                             "`client:`, `rendezvous:`, `post_docking:`, `separation:`); kept for old commands")
    parser.add_argument("--no_moving_dock", action="store_true",
                        help="Dock to a satellite at rest (the client is not released)")
    parser.add_argument("--no_mrv_approach", action="store_true",
                        help="Skip the MRV rendezvous phase (folded arm -> two translation legs -> arm deploy) "
                             "and start with the arm at the observation pose, as before (config `mrv:`)")
    parser.add_argument("--mrv_approach", action="store_true",
                        help="Force the MRV rendezvous phase on even if `mrv.enabled` is false in the config")
    parser.add_argument("--start_yaw_deg", type=float, default=None, metavar="DEG",
                        help="Start the arm swung DEG degrees in azimuth (about its base axis) so it has to search for the MEP")
    parser.add_argument("--no_astrobee", action="store_true",
                        help="Leave out the Astrobee observation camera (default: on, config `astrobee:`; its image is "
                             "published on /astrobee/camera/image_raw when ROS 2 is on)")
    parser.add_argument("--nozzle_obstruction", action="store_true",
                        help="Float a visual-only piece of debris at the satellite's docking port: the Astrobee map "
                             "must report it and the mission must end in DOCKING_UNAVAILABLE before inserting the probe")
    parser.add_argument("--no_ros", action="store_true", help="Disable the ROS 2 interface (default: on)")
    parser.add_argument("--ros", action="store_true",
                        help="Enable the ROS 2 interface (default: on) (telemetry topics + cmd/start, cmd/abort, cmd/capture_enable; see config `ros:`)")
    parser.add_argument("--ros_wait_start", action="store_true", help="With --ros: hold the arm until a cmd/start message arrives")
    parser.add_argument("--start_paused", action="store_true", help="Wait for Play in the toolbar before starting")
    parser.add_argument("--kit_args", type=str, default=None,
                        help=f'Extra Kit arguments. GUI default: "--ext-folder {isaac_path}/apps --enable isaacsim.exp.base"')
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides for the env config")
    args = parser.parse_args()
    if args.kit_args is None and not args.headless:
        args.kit_args = f"--ext-folder {isaac_path}/apps --enable isaacsim.exp.base"
    return args


def ensure_ros2_env(args):
    """The Isaac Sim ROS 2 bridge libraries must be on LD_LIBRARY_PATH before the process
    starts (setting it later does not affect the dynamic loader): re-exec once with the
    environment the bridge needs."""
    ros_off = args.no_ros or any(x.replace(" ", "") == "ros.enabled=false" for x in args.sets)
    wants_ros = args.ros or args.ros_wait_start or not ros_off
    if not wants_ros or os.environ.get("MRV_ROS_ENV_READY") == "1":
        return
    isaac_path = os.environ.get("ISAAC_PATH", os.path.expanduser("~/isaac-sim"))
    distro = os.environ.get("ROS_DISTRO") or "jazzy"
    bridge = os.path.join(isaac_path, "exts", "isaacsim.ros2.bridge", distro)
    lib, py = os.path.join(bridge, "lib"), os.path.join(bridge, "rclpy")
    if not os.path.isdir(lib) or not os.path.isdir(py):
        sys.exit(f"[ROS] bridge libraries not found: {bridge} (set ROS_DISTRO to humble / jazzy)")

    # A sourced /opt/ros (Python 3.12) or a separately built rclpy overlay must not mix with the
    # bridge's Python 3.11 rclpy + generated messages (rcl_interfaces ParameterEvent assertion):
    # use the bridge exclusively, first on both search paths.
    def external_ros(p):
        return p.startswith("/opt/ros/") or "ros_jazzy_py311" in p or "python3.12" in p

    for var, first in (("LD_LIBRARY_PATH", lib), ("PYTHONPATH", py), ("AMENT_PREFIX_PATH", None)):
        paths = [x for x in os.environ.get(var, "").split(":") if x and x != first and not external_ros(x)]
        os.environ[var] = ":".join(([first] if first else []) + paths)
        if not os.environ[var]:
            del os.environ[var]
    os.environ["ROS_DISTRO"] = distro
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ["MRV_ROS_ENV_READY"] = "1"
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


def main():
    args = parse_args()
    ensure_ros2_env(args)

    from srb.core.app import AppLauncher
    from srb.utils.path import SRB_APPS_DIR, SRB_LOGS_DIR
    from srb.utils.process import install_session_signal_handlers, warn_about_stale_sessions

    warn_about_stale_sessions()
    launcher_kwargs = dict(
        headless=args.headless,
        enable_cameras=True,
        experience=SRB_APPS_DIR.joinpath(f"srb.{'headless.' if args.headless else ''}rendering.kit"),
    )
    if args.kit_args:
        launcher_kwargs["kit_args"] = args.kit_args
    launcher = AppLauncher(**launcher_kwargs)
    install_session_signal_handlers()

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.util.debug_draw")

    import gymnasium

    import srb.tasks  # noqa: F401  (registers srb/debris_capture_vision)
    from srb.tasks.manipulation.debris_capture.vision import DEFAULT_CONFIG_PATH
    from srb.tasks.manipulation.debris_capture.vision_capture_demo import VisionCaptureDemo, default_out_dir
    from srb.utils.hydra.sim import hydra_task_config

    env_id = "srb/debris_capture_vision"
    hydra_dir = SRB_LOGS_DIR.joinpath("vision_capture", "hydra", datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        hydra_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        hydra_dir = Path(tempfile.mkdtemp(prefix="vision_capture_"))
    sys.argv = [sys.argv[0], "env.robot=canadarm3", *args.overrides, f"hydra.run.dir={hydra_dir}", "hydra.output_subdir=null"]

    sets = list(args.sets)
    if args.start_yaw_deg is not None:
        sets.append(f"approach.start_yaw_offset_deg={args.start_yaw_deg}")
    if args.no_mrv_approach and args.mrv_approach:
        sys.exit("[ARGS] --mrv_approach and --no_mrv_approach are mutually exclusive")
    if args.no_mrv_approach:
        sets.append("mrv.enabled=false")
    if args.mrv_approach:
        sets.append("mrv.enabled=true")
    if args.dock and args.no_dock:
        sys.exit("[ARGS] --dock and --no_dock are mutually exclusive")
    if args.moving_dock and (args.no_dock or args.no_moving_dock):
        sys.exit("[ARGS] --moving_dock and --no_dock / --no_moving_dock are mutually exclusive")
    if args.no_dock:
        sets.append("docking.enabled=false")
    elif args.dock or args.dock_only or args.moving_dock:
        sets.append("docking.enabled=true")
    # Moving client: on by default whenever the docking phase runs (the static scenario is
    # capture only, so it never applies there). An explicit --set of either key wins.
    user_sets = [x.replace(" ", "") for x in args.sets]
    user_chose = any(x.startswith("client.release_enabled=") for x in user_sets) or "docking.enabled=false" in user_sets
    if args.moving_dock or not (args.no_dock or args.no_moving_dock or args.scenario == "static" or user_chose):
        sets.append("client.release_enabled=true")
    if args.dock_only:
        sets.append("docking.skip_capture=true")
        # The docking-only path attaches the MEP at its nominal grasp pose in `start()`,
        # so there is no capture to approach and the rendezvous phase has no purpose
        sets.append("mrv.enabled=false")
    if args.no_astrobee:
        sets.append("astrobee.enabled=false")
    if args.nozzle_obstruction:
        if args.no_astrobee:
            sys.exit("[ARGS] --nozzle_obstruction needs the Astrobee (drop --no_astrobee)")
        sets.append("astrobee.map.test_obstruction=true")
    if args.no_ros:
        sets.append("ros.enabled=false")
    elif args.ros or args.ros_wait_start:
        sets.append("ros.enabled=true")
    if args.ros_wait_start:
        sets.append("ros.require_start_cmd=true")
    if args.scenario == "static":
        # Test 1: the MEP is at rest (no drift, no rotation)
        sets.append("mep.linear_velocity_mps=0.0")
        sets.append("mep.motion_mode=translation_only")
        # ... capture only, from the observation pose (the full-pipeline defaults do not apply)
        sets.append("docking.enabled=false")
        sets.append("approach.start_yaw_offset_deg=0.0")
        # ... and it is a pose-accuracy measurement, not a demo: the arm starts at the
        # observation pose as it always has (--mrv_approach forces the phase back on)
        if not args.mrv_approach:
            sets.append("mrv.enabled=false")
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"ok": False}

    @hydra_task_config(task_name=env_id)
    def run(env_cfg, agent_cfg=None):
        env_cfg.vision_config_path = str(Path(args.config).resolve()) if args.config else DEFAULT_CONFIG_PATH.as_posix()
        env_cfg.vision_overrides = tuple(sets)
        env_cfg.apply_vision_config()
        env = gymnasium.make(env_id, cfg=env_cfg)
        env.reset()
        sim = env.unwrapped.sim
        if not args.headless:
            sim.set_camera_view(eye=(-1.0, -9.0, 7.0), target=(5.0, -2.7, 3.0))
        if args.start_paused:
            sim.pause()
            print("[DEMO] PAUSED - press Play in the Isaac Sim toolbar to start", flush=True)
        tag = args.tag or ("full_6dof" if args.scenario == "dynamic" else args.scenario)
        csv_path = out_dir / f"{tag}_metrics.csv"
        demo = VisionCaptureDemo(env, launcher.app, args.scenario, args.headless, out_dir, csv_path, label=tag)
        results = demo.run()
        result_path = out_dir / f"{tag}_result.json"
        demo.results.metrics["command"] = " ".join([Path(sys.executable).name, *os.sys.orig_argv[1:]]) if hasattr(os.sys, "orig_argv") else ""
        demo.results.metrics["overrides"] = sets
        demo.write_json(result_path)
        print("[SUMMARY] ------------------------------------------------", flush=True)
        for name, c in results.checks.items():
            print(f"[SUMMARY] {'PASS' if c['pass'] else 'FAIL'}  {name}", flush=True)
        n_ok = sum(1 for c in results.checks.values() if c["pass"])
        print(f"[SUMMARY] {n_ok}/{len(results.checks)} checks passed, final state {results.final_state}"
              f"{'  (' + results.failure + ')' if results.failure else ''}", flush=True)
        print(f"[SUMMARY] results: {result_path}", flush=True)
        print(f"[SUMMARY] telemetry: {csv_path}", flush=True)
        state["ok"] = n_ok == len(results.checks)
        # Keep the scene up after the run reached SUCCESS, even if a report-only check
        # failed (the exit code below still carries every check)
        # SUCCESS, or stopped by the Astrobee (docking unavailable): results are already
        # written; keep the scene up (Astrobee still scanning) until the window closes
        if not args.headless and results.final_state in ("SUCCESS", "DOCKING_UNAVAILABLE") and not args.exit_when_done:
            demo.idle()
        demo.close_astrobee()
        env.close()

    run()
    ## NOTE: `SimulationApp.close()` ends the process itself with exit code 0
    ## (`unload_all_plugins`), so a `sys.exit` after it never runs. The results are
    ## already written; leave with the real status instead.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if state["ok"] else 1)


if __name__ == "__main__":
    main()
