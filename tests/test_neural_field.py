# SPDX-License-Identifier: GPL-3.0-only
"""Image-program behavior, continuation, numerical and hostile-input tests."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import numpy as np
from PIL import Image
import graph_vm as vm
import neural_field as nf


class ImageGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job=nf.prepare_image_job(size=8,steps=1)
        cls.program=cls.job['program'];cls.model=cls.job['model']

    def state(self,n=8):
        state=np.zeros((n,n),dtype=np.uint8);state[n//2,n//2]=1
        return state

    def test_all_32_or_and_neighborhoods(self):
        for model,expected in [(self.model,lambda bits:bits!=0),({'weights':[1.0]*5,'bias':-4.5},lambda bits:bits==31)]:
            for bits in range(32):
                state=np.zeros((8,8),dtype=np.uint8)
                for i,p in enumerate([(3,3),(2,3),(4,3),(3,2),(3,4)]):state[p]=(bits>>i)&1
                result,_=vm.checked_step(self.program,model,state)
                self.assertEqual(int(result[3,3]),int(expected(bits)))

    def test_model_mutation_changes_tiff_and_gif_computation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,model,count in [('or',self.model,5),('and',{'weights':[1.0]*5,'bias':-4.5},0)]:
                source=root/name;source.mkdir()
                nf.save_carriers(source,[nf.frame(name)],nf.make_image_job(self.state(),self.program,model))
                for ext in ('tiff','gif'):
                    receipt=nf.execute_carrier(source/f'processing.{ext}',root/f'{name}-{ext}')
                    self.assertEqual(receipt['result']['active_cells'],count)
                    self.assertTrue(receipt['result']['model_from_image'])

    def test_program_mutation_changes_tiff_and_gif_computation(self):
        changed=copy.deepcopy(self.program);changed['code'][1][3]=[[0,0]]*5
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,program,count in [('spread',self.program,5),('center',changed,1)]:
                source=root/name;source.mkdir()
                nf.save_carriers(source,[nf.frame(name)],nf.make_image_job(self.state(),program,self.model))
                for ext in ('tiff','gif'):
                    receipt=nf.execute_carrier(source/f'processing.{ext}',root/f'{name}-{ext}')
                    self.assertEqual(receipt['result']['active_cells'],count)
                    self.assertEqual(receipt['result']['program_sha256'],nf.digest(nf.canonical(program)))

    def test_image_state_controls_output(self):
        for state,expected in [(np.zeros((8,8),dtype=np.uint8),0),(self.state(),5)]:
            job=nf.make_image_job(state,self.program,self.model)
            decoded=nf.decode_payload(nf.encode_payload(nf.frame('state'),job))
            result,_=vm.checked_step(decoded['program'],decoded['model'],np.asarray(decoded['state'],dtype=np.uint8))
            self.assertEqual(int(result.sum()),expected)

    def test_refreshed_tiff_and_gif_continue_saved_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);nf.demo(root/'first',self.job)
            for ext in ('tiff','gif'):
                image=root/'first'/f'processing.{ext}';job,_=nf.read_carrier(image)
                self.assertEqual(job['tick'],1);self.assertEqual(sum(map(sum,job['state'])),5)
                result=nf.execute_carrier(image,root/ext)['result']
                self.assertEqual((result['initial_tick'],result['final_tick'],result['active_cells']),(1,2,13))

    def test_execute_never_retrains(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'in.tiff'
            nf.encode_payload(nf.frame('input'),self.job).save(source)
            with mock.patch.object(nf,'train_local_or',side_effect=AssertionError('no retraining')):
                self.assertEqual(nf.execute_carrier(source,root/'out')['result']['active_cells'],5)

    def test_generic_one_weight_graph_and_multiply(self):
        program={'schema':'nnf-graph/1','code':[['STATE',0],['CONST',1,1.0],['MUL',2,0,1],['GATHER',3,2,[[0,0]]],
            ['PARAM',4,'weights'],['DOT',5,3,4],['PARAM',6,'bias'],['ADD',7,5,6],['SIGMOID',8,7],
            ['CONST',9,0.5],['GE',10,8,9],['RETURN',10,8]]}
        model={'weights':[1.0],'bias':-0.5}
        result,_=vm.checked_step(program,model,self.state());np.testing.assert_array_equal(result,self.state())
        with tempfile.TemporaryDirectory() as temp:
            receipt=nf.demo(Path(temp)/'one',nf.make_image_job(self.state(),program,model))
            self.assertEqual(receipt['result']['active_cells'],1)

    def test_independent_manhattan_oracle_and_zero_boundary(self):
        state=np.zeros((16,16),dtype=np.uint8);state[0,0]=1;yy,xx=np.indices(state.shape)
        for tick in range(1,9):
            state,_=vm.checked_step(self.program,self.model,state)
            np.testing.assert_array_equal(state,((yy+xx)<=tick).astype(np.uint8))

    def test_no_program_model_or_state_mutation(self):
        state=self.state();p=copy.deepcopy(self.program);m=copy.deepcopy(self.model)
        result,prob=vm.checked_step(p,m,state);result[:]=0;prob[:]=0
        np.testing.assert_array_equal(state,self.state());self.assertEqual(p,self.program);self.assertEqual(m,self.model)

    def test_unknown_ops_registers_offsets_and_models_rejected(self):
        for ins in [['IMPORT',0,'os'],['STATE',16],['DOT',3,15,0],['GATHER',2,0,[[100,0]]]]:
            p=copy.deepcopy(self.program);p['code'][1]=ins
            with self.assertRaises(vm.GraphError):vm.execute(p,self.model,self.state())
        for model in [{'weights':[float('nan')]*5,'bias':0},{'weights':[1.0]*10,'bias':0},
                      {'weights':[1.0]*5,'bias':10**2000},{'weights':[True]*5,'bias':0}]:
            with self.assertRaises(vm.GraphError):vm.execute(self.program,model,self.state())
        with self.assertRaises(vm.GraphError):vm.execute(self.program,self.model,np.zeros((65,65),dtype=np.uint8))

    def test_extreme_logits_remain_finite(self):
        for bias in (-1e6,1e6):
            with np.errstate(over='raise',invalid='raise'):
                _,p=vm.checked_step(self.program,{'weights':[1e6]*5,'bias':bias},self.state())
                self.assertTrue(np.isfinite(p).all())

    def test_direct_reference_entry_point_rejects_growth_shapes_and_invalid_output(self):
        growth={'schema':'nnf-graph/1','code':[['CONST',0,1000000],['MUL',0,0,0],['MUL',0,0,0],['RETURN',0,0]]}
        nonbinary={'schema':'nnf-graph/1','code':[['STATE',0],['CONST',1,2],['MUL',2,0,1],['RETURN',2,0]]}
        for program in (growth,nonbinary):
            with self.assertRaises(vm.GraphError):vm.execute_reference(program,self.model,self.state())
        with self.assertRaises(vm.GraphError):vm.execute_reference(self.program,{'weights':[1.0],'bias':0},self.state())

    def test_checkpoint_carries_program_and_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=copy.deepcopy(self.job);job['steps']=6
            first=nf.demo(root/'first',job);self.assertTrue((root/'first/checkpoint/program.json').is_file())
            resumed=nf.resume_checkpoint(root/'first/checkpoint',3,root/'resume')
            self.assertEqual(first['result']['state_roots'][-1],resumed['result']['state_roots'][-1])
            self.assertEqual((root/'first/accepted_pixel160.bin').read_bytes(),(root/'resume/accepted_pixel160.bin').read_bytes())

    def test_checkpoint_code_tamper_and_gray_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);nf.demo(root/'first',self.job);cp=root/'first/checkpoint'
            (cp/'program.json').write_text('{}')
            with self.assertRaisesRegex(nf.Rejected,'integrity'):nf.load_checkpoint(cp)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);nf.demo(root/'first',self.job);cp=root/'first/checkpoint'
            Image.fromarray(np.full((8,8),127,dtype=np.uint8)).save(cp/'state.tiff')
            manifest=json.loads((cp/'checkpoint.json').read_text());manifest['state_sha256']=nf.digest((cp/'state.tiff').read_bytes())
            nf.write_json(cp/'checkpoint.json',manifest)
            with self.assertRaisesRegex(nf.Rejected,'exactly 0 or 255'):nf.load_checkpoint(cp)

    def test_conflicting_and_corrupt_later_frames_rejected(self):
        first=nf.encode_payload(nf.frame('one'),self.job);job=copy.deepcopy(self.job);job['model']['bias']=-4.5
        damaged=first.copy();damaged.putpixel((0,nf.Y0),(127,127,127))
        for second in [nf.encode_payload(nf.frame('other'),job),damaged]:
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'bad.tiff';first.save(path,save_all=True,append_images=[second])
                with self.assertRaises(nf.Rejected):nf.read_carrier(path)

    def test_checksum_dimensions_file_and_frame_bounds(self):
        image=nf.encode_payload(nf.frame('test'),self.job);bit=44*8+8;x=bit%nf.WIDTH;y=nf.Y0+bit//nf.WIDTH
        v=255-image.getpixel((x,y))[0];image.putpixel((x,y),(v,v,v))
        with self.assertRaisesRegex(nf.Rejected,'checksum'):nf.decode_payload(image)
        with self.assertRaises(nf.Rejected):nf.encode_payload(Image.new('RGB',(1,1)),self.job)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'large';path.write_bytes(b'x')
            with mock.patch.object(nf,'MAX_FILE',0),self.assertRaises(nf.Rejected):nf.read_carrier(path)
            with self.assertRaises(nf.Rejected):nf.save_carriers(Path(temp),[image]*33,self.job)

    def test_bad_json_legacy_recipe_and_state_types_rejected(self):
        for raw in [b'{"x":1,"x":2}',b'NaN',b'{"x":Infinity}',b'['*2000]:
            with self.assertRaises(nf.Rejected):nf.strict_json(raw)
        with self.assertRaises(nf.Rejected):nf.validate_job(nf.DEFAULT_PREPARATION)
        for key,value in [('steps',True),('steps',25),('tick',-1),('state',[[1]*65]*65)]:
            job=copy.deepcopy(self.job);job[key]=value
            with self.assertRaises(nf.Rejected):nf.validate_job(job)
        job=copy.deepcopy(self.job);job['state'][0][0]=True
        with self.assertRaises(nf.Rejected):nf.validate_job(job)

    def test_maximum_grid_fits_payload(self):
        job=nf.make_image_job(self.state(64),self.program,self.model)
        self.assertLess(len(nf.canonical(job)),nf.MAX_PAYLOAD)
        self.assertEqual(nf.decode_payload(nf.encode_payload(nf.frame('64'),job)),job)

    def test_failed_run_preserves_source_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'in.tiff';nf.encode_payload(nf.frame('in'),self.job).save(source);original=source.read_bytes()
            with self.assertRaises(nf.Rejected):nf.execute_carrier(source,root)
            with mock.patch.object(nf,'run_neural',side_effect=nf.Rejected('forced')):
                with self.assertRaises(nf.Rejected):nf.execute_carrier(source,root/'failed')
            self.assertFalse((root/'failed').exists());self.assertEqual(source.read_bytes(),original)

    def test_pixel160_and_probability_validation(self):
        state=self.state();raw=nf.pixel160_state(state.astype(float),state,7);offset=(4*8+4)*20
        self.assertEqual(len(raw),8*8*20);self.assertEqual(struct.unpack('>5I',raw[offset:offset+20]),(0x3f800000,1,7,1,0))
        with self.assertRaises(nf.Rejected):nf.pixel160_state(np.full((8,8),np.nan),state,1)

    def test_relative_evidence_and_mssl_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);receipt=nf.demo(root/'demo',self.job);self.assertFalse(Path(receipt['source_path']).is_absolute())
            source=(root/'demo/evidence.mssl').read_text(encoding='utf-8');body=source[source.index('α≡'):source.rindex(',κ≡')]
            seal=json.loads(source[source.rindex(',κ≡')+3:source.rindex('\n}⇒⊤')])
            self.assertEqual(seal,'sha256:'+hashlib.sha256(body.encode()).hexdigest());self.assertIn('program_from_image',source)


if __name__=='__main__':unittest.main()
