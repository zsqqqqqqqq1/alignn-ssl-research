"""从正式配置选择训练流程；流程开关不参与旧模型的断点签名。"""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / 'configs'
MODES = ('learnable', 'zero', 'zero_direct')


def training_config(config):
    """剥离流程开关，保留所有实际训练参数；旧配置缺少开关时默认为自监督。"""
    result = copy.deepcopy(config)
    enabled = result.pop('self_supervised', True)
    if type(enabled) is not bool:
        raise ValueError('self_supervised must be a JSON Boolean: true or false')
    return enabled, result


def supervised_config(config):
    """复用同一配置的监督参数，删除只属于掩码预训练和 Z 投影的字段。"""
    _, result = training_config(config)
    result.pop('pretrain', None)
    result['model'].pop('z_dim', None)
    # 沿用现有稳定的数据加载方式，不创建易崩溃的 spawn 子进程。
    result['num_workers'] = 0
    return result


def select_experiment(argv, directory=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=('both', *MODES), default='both')
    args, _ = parser.parse_known_args(argv)
    directory = Path(directory) if directory is not None else CONFIGS
    names = ('learnable', 'zero') if args.mode == 'both' else (args.mode,)
    configs = [json.loads((directory / f'formal_{name}.json').read_text(encoding='utf-8'))
               for name in names]
    flags = [training_config(config)[0] for config in configs]
    if len(set(flags)) != 1:
        raise ValueError('--mode both requires matching self_supervised values in learnable and zero; '
                         'use --mode learnable or --mode zero to run one configuration')
    if flags[0]:
        return True, None
    controls = [supervised_config(config) for config in configs]
    if any(config != controls[0] for config in controls[1:]):
        raise ValueError('Supervised settings differ; select one configuration with --mode')
    return False, controls[0]


@contextmanager
def legacy_config_reads(launcher):
    """只在旧入口读取三份配置时剥离开关，维持原有协议和源码哈希。

    真实训练参数仍完整校验；不会修改磁盘文件，也不会放宽数据或权重校验。
    """
    original = launcher.read_json
    paths = {(launcher.CODE / 'configs' / f'formal_{name}.json').resolve() for name in MODES}

    def read(path):
        value = original(path)
        return training_config(value)[1] if Path(path).resolve() in paths else value

    launcher.read_json = read
    try:
        yield
    finally:
        launcher.read_json = original
