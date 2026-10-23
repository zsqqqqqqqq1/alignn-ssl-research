"""正式训练的简洁显示入口；计算仍交给原 run_formal.py。"""
import sys
from contextlib import nullcontext


def main():
    stable_loader = "--single-process-loader" in sys.argv
    if stable_loader:
        sys.argv[:] = [v for v in sys.argv if v != "--single-process-loader"]
    from terminal_display import TerminalDisplay, demo
    if sys.argv[1:] == ["--demo"]:
        demo()
        return
    # 直接读取所选 formal 配置中的开关，关闭时沿用该配置做原版监督训练。
    from experiment_options import select_experiment, legacy_config_reads
    enabled, config = select_experiment(sys.argv[1:])
    if not enabled:
        import supervised_run
        supervised_run.main(config)
        return
    # --demo 不导入训练栈；实际运行才加载原入口。
    import run_formal
    from ssl_alignn import runner
    from single_process_loader import single_process_loading
    if stable_loader:
        print("Loader override: num_workers=0; original experiment config/checkpoint remain intact.", flush=True)
    with legacy_config_reads(run_formal), (single_process_loading() if stable_loader else nullcontext()), TerminalDisplay().installed(run_formal, runner):
        run_formal.main()


# Windows/WSL 的 spawn 子进程会重新导入入口。必须保留此保护，
# 否则数据加载进程也可能再次启动训练。
if __name__ == "__main__":
    main()
