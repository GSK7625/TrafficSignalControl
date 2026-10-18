"""Content identity v1. Git metadata is trace information, never identity."""
from pathlib import Path
from dataclasses import asdict
from typing import Sequence
import hashlib
import json
import platform
import subprocess
import importlib.metadata
import xml.etree.ElementTree as ET

SCHEMA = 'stage05-provenance-v1'
METRIC_SCHEMA = '4.0.0'
PROTOCOL = 'stage04-schema4-v1'
ROOT = Path(__file__).resolve().parents[2]

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def hash_files(paths: Sequence[str], root: str) -> str:
    root=Path(root).resolve()
    rows=[]
    for path in paths:
        p=Path(path);p=(root/p).resolve() if not p.is_absolute() else p.resolve()
        relative=p.relative_to(root).as_posix()
        rows.append((relative,p.read_bytes()))
    h=hashlib.sha256()
    for name, content in sorted(rows):
        for blob in (name.encode('utf-8'),content):
            h.update(len(blob).to_bytes(8,'big'));h.update(blob)
    return h.hexdigest()

def input_files(config):
    config=Path(config).resolve();files={config}
    tree=ET.parse(config)
    for node in tree.findall('./input/*'):
        if node.tag.endswith(('-file','-files')):
            for value in node.get('value','').split(','):
                if value.strip():files.add((config.parent/value.strip()).resolve())
    # XML includes are inputs too; recurse without following output attributes.
    pending=list(files)
    while pending:
        p=pending.pop()
        if p.suffix in ('.xml','.sumocfg'):
            for node in ET.parse(p).iter():
                if node.tag.split('}')[-1] != 'include':
                    continue
                target=(p.parent/node.attrib['href']).resolve()
                if target not in files:files.add(target);pending.append(target)
    return sorted(files)

def environment(binary):
    binary=Path(binary).resolve()
    return dict(python=platform.python_version(),implementation=platform.python_implementation(),
        os=platform.system(),machine=platform.machine(),
        dependencies={name:importlib.metadata.version(name) for name in ('numpy','sumolib','traci','pandas','pyyaml','matplotlib')},
        sumo_version=subprocess.check_output([str(binary),'--version'],text=True).splitlines()[0],
        sumo_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())

def build_provenance(cfg, checkpoint_path=None, root=None):
    from src.config import get_scenario_config
    from src.evaluation.experiment_runner import resolve_effective_decision_interval
    from src.evaluation.evaluate_rl import load_checkpoint_env_params
    root=Path(root or ROOT).resolve()
    sim=get_scenario_config(cfg.scenario_name,data_dir=str(root/'data'),max_steps=cfg.steps)
    model=checkpoint_path or cfg.model_path
    if model is None and cfg.algorithm in ('Q-Learning','q_learning','qlearning'):
        best=root/'models/q_learning'/f'q_learning_{cfg.scenario_name}_best.json'
        final=root/'models/q_learning'/f'q_learning_{cfg.scenario_name}_final.json'
        model=best if best.is_file() else final if final.is_file() else best
    model=(root/model).resolve() if model and not Path(model).is_absolute() else Path(model).resolve() if model else None
    params=load_checkpoint_env_params(str(model)) if model else {}
    sources=[root/'main.py',*sorted((root/'src').rglob('*.py'))]
    inputs=input_files(sim.sumocfg_path)
    effective=asdict(cfg)
    for key in ('output_dir','model_path'):effective.pop(key)
    effective.update(decision_interval=resolve_effective_decision_interval(cfg,params),step_length=1.0,
        begin_time=sim.begin_time,end_time=sim.end_time,horizon_steps=cfg.steps if cfg.steps is not None else int(sim.end_time-sim.begin_time))
    identity=dict(schema=SCHEMA,metric_schema=METRIC_SCHEMA,protocol=PROTOCOL,effective_config=effective,
        source=hash_files([str(p) for p in sources],str(root)),inputs=hash_files([str(p) for p in inputs],str(root)),
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest() if model else None,environment=environment(sim.sumo_binary))
    cohort=json.loads(json.dumps(identity));cohort['effective_config'].pop('seed')
    try:
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True))
    except subprocess.CalledProcessError:commit=None;dirty=None
    return dict(schema=SCHEMA,identity=identity,run_fingerprint=digest(identity),cohort_fingerprint=digest(cohort),
        paths=dict(root=str(root),source=[str(p) for p in sources],inputs=[str(p) for p in inputs],model=str(model) if model else None,sumo=str(Path(sim.sumo_binary).resolve())),
        git=dict(commit=commit,dirty=dirty))

def reusable_record(record, expected):
    p=record.get('provenance') or {};identity=p.get('identity') or {}
    required={'schema','metric_schema','protocol','effective_config','source','inputs','model_sha256','environment'}
    return (record.get('status')=='SUCCESS' and record.get('is_valid') is True and record.get('data_validity')=='VALID'
        and record.get('metric_schema_version')==METRIC_SCHEMA and p.get('schema')==SCHEMA
        and required <= identity.keys() and identity.get('schema')==SCHEMA
        and {'python','implementation','os','machine','dependencies','sumo_version','sumo_sha256'} <= identity.get('environment',{}).keys()
        and {'seed','training_seed','decision_interval','step_length','horizon_steps','min_green','max_green','yellow_duration','all_red_duration'} <= identity.get('effective_config',{}).keys()
        and digest(identity)==p.get('run_fingerprint')==record.get('fingerprint')==expected)


def comparison_identity(provenance):
    """Shared code/data/environment/protocol required for baseline comparisons."""
    if not provenance:
        return None
    identity=provenance.get('identity',{})
    return {key:identity.get(key) for key in ('source','inputs','protocol','metric_schema','environment')}
