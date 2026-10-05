# SPDX-License-Identifier: GPL-3.0-only
"""Real .NET execution, image-only mutations, native/Python parity and isolation."""
import base64
import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image
import graph_vm
import native_carrier as nc
import neural_field as nf

ROOT=Path(__file__).resolve().parents[1]
DOTNET=os.environ.get('NNF_DOTNET') or shutil.which('dotnet')
PUBLISHED=os.environ.get('NNF_PLAYER_DIR')
PLAYER=Path(PUBLISHED)/'NeuralPlayer.exe' if PUBLISHED else ROOT/'dotnet/NeuralPlayer/bin/Release/net10.0-windows/NeuralPlayer.dll'
AVAILABLE=os.name=='nt' and PLAYER.is_file() and (PUBLISHED or DOTNET)

def unchecked(path,job):
    raw=nf.canonical(job);packet=nc.MAGIC+struct.pack('>I',len(raw))+bytes.fromhex(nf.digest(raw))+raw
    bits=np.unpackbits(np.frombuffer(packet,dtype=np.uint8));pixels=np.full((nf.HEIGHT,nf.WIDTH,3),255,dtype=np.uint8)
    pixels[nf.Y0:].reshape(-1,3)[:len(bits)]=bits[:,None]*255;Image.fromarray(pixels).save(path)

@unittest.skipUnless(AVAILABLE,'Build the Windows player or supply NNF_PLAYER_DIR')
class NativePlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job=nc.author(nf.prepare_image_job(size=8,steps=1))

    def launch(self,path,output=None,command=None,cwd=None,environment=None):
        command=(command or ([str(PLAYER)] if PUBLISHED else [DOTNET,str(PLAYER)]))+['--step',str(path)]
        if output is not None:command+=['--out',str(output)]
        return subprocess.run(command,capture_output=True,text=True,timeout=30,cwd=cwd,env=environment)

    def accepted(self,path,output=None,**kwargs):
        result=self.launch(path,output,**kwargs);self.assertEqual(result.returncode,0,result.stderr)
        receipt=json.loads(result.stdout);self.assertEqual(receipt['runtime_location'],'image pixels');return receipt

    def test_image_model_graph_mutation_and_exported_continuation_both_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cases={'or':copy.deepcopy(self.job),'and':copy.deepcopy(self.job),'center':copy.deepcopy(self.job)}
            cases['and']['model']={'weights':['1']*5,'bias':'-4.5'};cases['center']['program']['code'][1][3]=[[0,0]]*5
            for name,job in cases.items():
                for extension in ('tiff','gif'):
                    with self.subTest(name=name,extension=extension):
                        source=root/(name+'.'+extension);nc.encode(nf.frame(name),job).save(source)
                        output=root/(name+'-next.'+extension);receipt=self.accepted(source,output)
                        state,probability=nc.reference(job)
                        np.testing.assert_array_equal(receipt['value']['state'],state)
                        np.testing.assert_allclose(receipt['probabilities'],probability,rtol=1e-12,atol=1e-12)
                        self.assertEqual(receipt['active_cells'],{'or':5,'and':0,'center':1}[name])
                        self.assertEqual(nc.read(output),receipt['value'])
                        next_receipt=self.accepted(output);state,probability=nc.reference(receipt['value'])
                        np.testing.assert_array_equal(next_receipt['value']['state'],state)
                        np.testing.assert_allclose(next_receipt['probabilities'],probability,rtol=1e-12,atol=1e-12)
                        self.assertEqual(next_receipt['value']['tick'],2)

    def test_all_32_neighborhoods_and_random_numeric_parity(self):
        # Separate neighborhoods occupy disjoint 6x6 tiles in one 64x64 field.
        legacy=nf.prepare_image_job(size=64,steps=1);state=np.zeros((64,64),dtype=np.uint8)
        for bits in range(32):
            y,x=3+(bits//8)*7,3+(bits%8)*7
            for i,(dy,dx) in enumerate([(0,0),(-1,0),(1,0),(0,-1),(0,1)]):state[y+dy,x+dx]=(bits>>i)&1
        legacy['state']=state.tolist()
        rng=np.random.default_rng(312)
        models=[legacy['model'],{'weights':[1]*5,'bias':-4.5}]+[{'weights':rng.uniform(-4,4,5).tolist(),'bias':float(rng.uniform(-2,2))} for _ in range(6)]
        with tempfile.TemporaryDirectory() as temp:
            for i,model in enumerate(models):
                legacy['model']=model;job=nc.author(legacy);source=Path(temp)/f'parity-{i}.tiff';nc.encode(nf.frame('Parity'),job).save(source)
                result=self.accepted(source);expected,probability=nc.reference(job)
                np.testing.assert_array_equal(result['value']['state'],expected)
                np.testing.assert_allclose(result['probabilities'],probability,rtol=1e-12,atol=1e-12)

    def test_generic_graph_scalar_broadcast_numeric_comparisons_and_multistep(self):
        legacy=nf.prepare_image_job(size=8,steps=3)
        legacy['program']={'schema':'nnf-graph/1','code':[['STATE',0],['CONST',1,0.5],['GE',2,0,1],['ADD',3,2,2],
            ['CONST',4,1.5],['GE',5,3,4],['CONST',6,1],['MUL',7,5,6],['RETURN',7,0]]}
        with tempfile.TemporaryDirectory() as temp:
            job=nc.author(legacy);source=Path(temp)/'generic.tiff';nc.encode(nf.frame('Generic'),job).save(source)
            result=self.accepted(source);state,probability=nc.reference(job)
            np.testing.assert_array_equal(result['value']['state'],state);self.assertEqual(result['value']['tick'],3)
            np.testing.assert_allclose(result['probabilities'],probability,rtol=1e-12,atol=1e-12)

    def test_deployed_host_no_python_authoring_or_loose_interpreter(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);deployed=root/'player'
            if PUBLISHED:
                shutil.copytree(Path(PUBLISHED),deployed);command=[str(deployed/'NeuralPlayer.exe')]
            else:
                deployed.mkdir()
                for name in ('NeuralPlayer.dll','NeuralPlayer.deps.json','NeuralPlayer.runtimeconfig.json'):shutil.copyfile(PLAYER.parent/name,deployed/name)
                command=[str(Path(DOTNET).resolve()),str(deployed/'NeuralPlayer.dll')]
            self.assertFalse(list(deployed.rglob('*.py')));self.assertFalse(list(deployed.rglob('NeuralRuntime.dll')))
            self.assertFalse(list(deployed.rglob('runtime-payload.json')))
            source=deployed/'input.tiff';nc.encode(nf.frame('Isolated'),self.job).save(source);out=deployed/'next.gif'
            environment=os.environ.copy();environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
            environment['PATH']=str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32')
            result=self.accepted(source,out,command=command,cwd=deployed,environment=environment)
            source.unlink();again=self.accepted(out,command=command,cwd=deployed,environment=environment)
            self.assertEqual((result['active_cells'],again['active_cells']),(5,13))

    def test_hostile_program_model_runtime_and_corruption_preserve_source(self):
        variants=[]
        for edit in ('opcode','register','offset','nan','overflow','dimensions','steps','runtime'):
            job=copy.deepcopy(self.job)
            if edit=='opcode':job['program']['code'][0]=['IMPORT',0,'os']
            elif edit=='register':job['program']['code'][0]=['STATE',16]
            elif edit=='offset':job['program']['code'][1][3]=[[100,0]]
            elif edit=='nan':job['model']['bias']='NaN'
            elif edit=='overflow':job['program']['code']=[['CONST',0,'1000000'],['MUL',0,0,0],['MUL',0,0,0],['RETURN',0,0]]
            elif edit=='dimensions':job['state']=[[0]*65 for _ in range(65)]
            elif edit=='steps':job['steps']=25
            else:
                raw=b'unknown interpreter';job['runtime']={'format':'dotnet-il/gzip-base64','sha256':nf.digest(raw),'data':base64.b64encode(gzip.compress(raw,mtime=0)).decode()}
            variants.append((edit,job))
        with tempfile.TemporaryDirectory() as temp:
            for name,job in variants:
                source=Path(temp)/(name+'.tiff');unchecked(source,job);before=source.read_bytes();output=Path(temp)/(name+'-out.gif')
                result=self.launch(source,output)
                self.assertEqual(result.returncode,1,result.stderr);self.assertIn('Carrier rejected:',result.stderr)
                self.assertFalse(output.exists());self.assertEqual(source.read_bytes(),before)
            source=Path(temp)/'corrupt.tiff';image=nc.encode(nf.frame('Corruption'),self.job);image.putpixel((0,nf.Y0),(42,42,42));image.save(source)
            self.assertEqual(self.launch(source).returncode,1)

    def test_conflicting_frames_dimensions_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);first=nc.encode(nf.frame('First'),self.job);other=copy.deepcopy(self.job);other['tick']=1
            for name,second in [('conflict',nc.encode(nf.frame('Second'),other)),('dimension',Image.new('RGB',(1,1)))]:
                path=root/(name+'.tiff');first.save(path,save_all=True,append_images=[second]);result=self.launch(path)
                self.assertEqual(result.returncode,1,result.stderr)
            path=root/'source.tiff';first.save(path);before=path.read_bytes();result=self.launch(path,path)
            self.assertEqual(result.returncode,1);self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
