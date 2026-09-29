"""CPU-only local checkpoint inspection. No model execution or checkpoint writes."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def deny(*args, **kwargs):
    raise RuntimeError('F11_FORBIDDEN_OPERATION')


def audit_hook(event, args):
    if event == 'open':
        if str(args[0]).lower() == os.devnull.lower():
            return  # Null device is not a persistent file; dill probes its IO type.
        mode = args[1] or ''
        flags = args[2] or 0
        if any(x in str(mode) for x in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            deny()
    if event == 'os.mkdir' and os.path.isdir(args[0]):
        return  # mkdir(exist_ok=True) on an existing directory cannot change it.
    if event in {'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.truncate', 'subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'}:
        deny()


def main():
    import json
    import zipfile
    from pathlib import Path
    import tempfile
    # Avoid tempfile's write-probe during dependency imports; all actual writes remain denied.
    tempfile.tempdir = str(Path(os.environ['TEMP']).resolve())
    path = Path(sys.argv[1]).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        print(json.dumps({'deserialization_attempted': False, 'integrity': 'INVALID_ZERO_BYTE_FILE'}))
        return
    sys.addaudithook(audit_hook)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError('ZIP_CRC_FAILURE:' + bad)
    import torch
    torch.set_num_threads(1)
    torch.cuda._lazy_init = deny
    torch.nn.Module.__call__ = deny
    torch.nn.Module._call_impl = deny
    torch.Tensor.backward = deny
    torch.autograd.backward = deny
    torch.save = deny
    torch.optim.Optimizer.step = deny
    for cls in vars(torch.optim).values():
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
            cls.step = deny
    # These are locally produced, hash-inventoried artifacts; never accept arbitrary remote pickle.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError('INVALID_CHECKPOINT_STRUCTURE')
    seen = set()
    tensors = {}

    def visit(value, name):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            tensors[name] = {'elements': value.numel(), 'dtype': str(value.dtype),
                             'cpu': value.device.type == 'cpu', 'finite': bool(torch.isfinite(value).all())}
        elif isinstance(value, torch.nn.Module):
            visit(value.__dict__, name)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, name + '/' + str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, name + '/' + str(index))

    visit(checkpoint, 'checkpoint')
    def field(*names):
        return any(checkpoint.get(name) is not None for name in names)
    metadata = {key: checkpoint.get(key) for key in ['epoch', 'best_fitness', 'updates', 'date', 'version']}
    args = checkpoint.get('train_args') or {}
    metadata['training_args'] = {key: args.get(key) for key in ['imgsz', 'batch', 'epochs', 'seed', 'optimizer', 'amp', 'resume', 'task']}
    states = {'model': field('model'), 'EMA': field('ema'), 'optimizer': field('optimizer'),
              'scheduler': field('scheduler', 'lr_scheduler'), 'GradScaler': field('scaler', 'grad_scaler'),
              'training_args': field('train_args'), 'update_counter': field('updates'),
              'RNG': field('rng_state', 'rng_states', 'random_state'), 'sampler': field('sampler', 'sampler_state'),
              'augmentation': field('augmentation_state', 'augmentation_rng_state')}
    nonfinite = {key: val for key, val in tensors.items() if not val['finite']}
    weight_tensors = [val for key, val in tensors.items() if key.startswith(('checkpoint/model/', 'checkpoint/ema/'))]
    weight_ok = bool(weight_tensors) and all(val['finite'] and val['cpu'] for val in weight_tensors)
    print(json.dumps({'container_readable': True, 'zip_crc_valid': True, 'structure': sorted(checkpoint),
                      'deserialization_attempted': True, 'metadata': metadata, 'states_present': states,
                      'tensor_count': len(tensors), 'all_tensors_cpu': all(v['cpu'] for v in tensors.values()),
                      'all_stored_tensors_finite': not nonfinite, 'nonfinite_tensors': nonfinite,
                      'model_or_ema_weights_finite': weight_ok,
                      'integrity': 'VALID' if weight_ok and not nonfinite else 'INVALID',
                      'forward': False, 'inference': False, 'optimizer_step': False, 'save': False,
                      'gpu_initialized': torch.cuda.is_initialized()}, allow_nan=False))


if __name__ == '__main__':
    main()
