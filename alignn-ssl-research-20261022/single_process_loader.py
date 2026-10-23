"""只替换数据读取的执行方式，保留已有训练协议与完整断点。

原有配置锁定 num_workers=4；直接改 JSON 会使断点拒绝恢复。
此适配器明确记录实际 num_workers=0，不修改配置、随机种子、样本顺序、
掩码、模型或优化器。代价是取消并行预取，读取速度可能变化。
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path


@contextmanager
def single_process_loading():
    from ssl_alignn.data import CrystalData
    original = CrystalData.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        requested = self.num_workers
        self.num_workers = 0
        # DataLoader 仍使用原来的独立 RNG、collate、批次及 pin_memory 设置。
        # 唯一改变是不创建 spawn 子进程，从而避开该子进程启动时的管道阻塞。
        record = dict(time_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                      dataset=str(self.root.resolve()), supervised=self.targets is not None,
                      configured_num_workers=requested, effective_num_workers=0,
                      reason='explicit single-process loader recovery adapter')
        audit = Path(self.root).parent / 'loader_runtime.jsonl'
        with audit.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(f'Data loading: configured workers={requested}, effective workers=0 '
              '(single process; runtime override logged).', flush=True)

    CrystalData.__init__ = initialize
    try:
        yield
    finally:
        CrystalData.__init__ = original
