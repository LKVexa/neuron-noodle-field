# SPDX-License-Identifier: GPL-3.0-only
"""Author/inspect the native image profile; Python is an independent reference only."""
import argparse
import base64
import copy
import gzip
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw

import graph_vm
import neural_field as nf

MAGIC=b'NNFIELD3'
MAX_PAYLOAD=24000
ROOT=Path(__file__).resolve().parent


def reference_parts(job):
    program=copy.deepcopy(job['program'])
    if program.get('schema')!='nnf-graph/2':raise nf.Rejected('Unsupported native graph')
    program['schema']='nnf-graph/1'
    def number(value):
        if type(value) is not str or len(value)>64:raise nf.Rejected('Expected bounded decimal string')
        try:return graph_vm.finite_number(float(value))
        except (ValueError,OverflowError) as exc:raise nf.Rejected('Invalid decimal number') from exc
    for instruction in program.get('code',[]):
        if isinstance(instruction,list) and instruction and instruction[0]=='CONST' and len(instruction)==3:
            instruction[2]=number(instruction[2])
    model=job['model']
    if type(model) is not dict or set(model)!={'weights','bias'} or type(model['weights']) is not list:raise nf.Rejected('Invalid native model')
    model={'weights':[number(x) for x in model['weights']],'bias':number(model['bias'])}
    return program,model


def validate(job):
    if type(job) is not dict or set(job)!={'schema','program','model','state','tick','steps','runtime'} or job['schema']!='nnf-native/1':
        raise nf.Rejected('Unsupported native image profile')
    program,model=reference_parts(job)
    nf.validate_job({'schema':'nnf-image/1','kind':'neural-graph','program':program,'model':model,'state':job['state'],'tick':job['tick'],'steps':job['steps']})
    runtime=job['runtime']
    if type(runtime) is not dict or set(runtime)!={'format','sha256','data'} or runtime['format']!='dotnet-il/gzip-base64' or type(runtime['data']) is not str or len(runtime['data'])>20000:
        raise nf.Rejected('Invalid image runtime')
    try:
        packed=base64.b64decode(runtime['data'],validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:module=stream.read(131073)
        if not 1<=len(module)<=131072 or nf.digest(module)!=runtime['sha256']:raise nf.Rejected('Runtime hash/size mismatch')
    except (ValueError,OSError,EOFError) as exc:raise nf.Rejected('Invalid runtime bytes') from exc
    if len(nf.canonical(job))>MAX_PAYLOAD:raise nf.Rejected('Native payload limit')
    return job


def author(legacy_job,runtime=None):
    nf.validate_job(legacy_job)
    program=copy.deepcopy(legacy_job['program']);program['schema']='nnf-graph/2'
    for instruction in program['code']:
        if instruction[0]=='CONST':instruction[2]=repr(float(instruction[2]))
    model={'weights':[repr(float(x)) for x in legacy_job['model']['weights']], 'bias':repr(float(legacy_job['model']['bias']))}
    if runtime is None:runtime=nf.strict_json((ROOT/'runtime-payload.json').read_bytes())
    return validate({'schema':'nnf-native/1','program':program,'model':model,'state':copy.deepcopy(legacy_job['state']),
        'tick':legacy_job['tick'],'steps':legacy_job['steps'],'runtime':copy.deepcopy(runtime)})


def encode(image,job):
    validate(job)
    if image.size!=(nf.WIDTH,nf.HEIGHT):raise nf.Rejected('Native carrier dimensions')
    raw=nf.canonical(job);packet=MAGIC+struct.pack('>I',len(raw))+bytes.fromhex(nf.digest(raw))+raw
    bits=np.unpackbits(np.frombuffer(packet,dtype=np.uint8))
    pixels=np.asarray(image.convert('RGB')).copy();stripe=pixels[nf.Y0:].reshape(-1,3)
    stripe[:]=255;stripe[:len(bits)]=bits[:,None]*255
    return Image.fromarray(pixels)


def decode(image):
    if image.size!=(nf.WIDTH,nf.HEIGHT):raise nf.Rejected('Native carrier dimensions')
    stripe=np.asarray(image.convert('RGB'))[nf.Y0:].reshape(-1,3)
    def take(length):
        cells=stripe[:length*8]
        if len(cells)!=length*8 or not np.all((cells==0)|(cells==255)) or not np.all(cells==cells[:,:1]):raise nf.Rejected('Damaged payload pixels')
        return np.packbits((cells[:,0]//255).astype(np.uint8)).tobytes()
    header=take(44)
    if header[:8]!=MAGIC:raise nf.Rejected('Unsupported native raster profile')
    size=struct.unpack('>I',header[8:12])[0]
    if not 1<=size<=MAX_PAYLOAD:raise nf.Rejected('Payload size limit')
    raw=take(44+size)[44:]
    if bytes.fromhex(nf.digest(raw))!=header[12:]:raise nf.Rejected('Payload checksum mismatch')
    job=nf.strict_json(raw)
    if nf.canonical(job)!=raw:raise nf.Rejected('Noncanonical payload')
    return validate(job)


def read(path):
    data=nf.bounded_bytes(path,nf.MAX_FILE)
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ('TIFF','GIF','PNG') or not 1<=image.n_frames<=32:raise nf.Rejected('Unsupported or excessive image')
        expected=None
        for frame in range(image.n_frames):
            image.seek(frame);job=decode(image)
            if expected is not None and job!=expected:raise nf.Rejected('Conflicting frame payloads')
            expected=job
        return expected


def reference(job):
    validate(job);program,model=reference_parts(job);state=np.asarray(job['state'],dtype=np.uint8)
    for _ in range(job['steps']):state,probabilities=graph_vm.checked_step(program,model,state)
    return state,probabilities


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    legacy=nf.prepare_image_job(size=32,steps=1)
    variants={'or':legacy,'and':copy.deepcopy(legacy),'center':copy.deepcopy(legacy)}
    variants['and']['model']={'weights':[1.0]*5,'bias':-4.5}
    variants['center']['program']['code'][1][3]=[[0,0]]*5
    report={'version':'0.4.0','profile':'nnf-native/1','cases':{}}
    for name,source in variants.items():
        job=author(source);state,probability=reference(job)
        visible=nf.frame('Runtime in pixels: '+name,'Open this image in NeuralPlayer; execute, run or export its next state')
        draw=ImageDraw.Draw(visible);n=len(job['state']);cell=280/n
        for y,row in enumerate(job['state']):
            for x,value in enumerate(row):
                draw.rectangle((35+x*cell,158+y*cell,35+(x+1)*cell,158+(y+1)*cell),fill='#68eeae' if value else '#191970')
        for i,line in enumerate(['Current field: one active seed','Tick 0  |  '+str(n)+' x '+str(n),'Interpreter, graph, model and state','are encoded in the black/white pixels.','Right: execute  |  Space: run/pause','O: change image  |  E: export next state']):
            draw.text((375,170+i*40),line,font=nf.font(21),fill='white' if i<2 else '#aec0d7')
        image=encode(visible,job)
        paths={}
        for ext in ('tiff','gif'):
            path=output/(name+'.'+ext);image.save(path)
            if read(path)!=job:raise nf.Rejected('Native example roundtrip failed')
            paths[ext]={'sha256':nf.digest(path.read_bytes())}
        report['cases'][name]={'formats':paths,'expected_active_cells':int(state.sum()),'runtime_sha256':job['runtime']['sha256'],
            'program_sha256':nf.digest(nf.canonical(job['program'])),'model_sha256':nf.digest(nf.canonical(job['model']))}
    nf.write_json(output/'reference-evidence.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('examples-native'));args=parser.parse_args()
    print(json.dumps(prepare(args.output),indent=2))
