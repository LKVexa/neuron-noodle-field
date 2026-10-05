# SPDX-License-Identifier: GPL-3.0-only
"""Author example image programs and record behavioral changes without editing the executor."""
import argparse
import copy
from pathlib import Path
import tempfile

import graph_vm
import neural_field as nf


def prepare(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    original=nf.prepare_image_job(size=8,steps=1)
    cases={'or':original,'and':copy.deepcopy(original),'center':copy.deepcopy(original)}
    cases['and']['model']={'weights':[1.0]*5,'bias':-4.5}
    cases['center']['program']['code'][1][3]=[[0,0]]*5
    expected={'or':5,'and':0,'center':1}
    report={'version':nf.VERSION,'executor_sha256':nf.digest(Path(graph_vm.__file__).read_bytes()),'cases':{}}
    with tempfile.TemporaryDirectory(prefix='nnf-example-verification-') as temp:
        work=Path(temp)
        for name,job in cases.items():
            source=work/name;source.mkdir()
            nf.save_carriers(source,[nf.frame('Image program: '+name,'Change the carried model or graph to change the computation')],job)
            record={'program_sha256':nf.digest(nf.canonical(job['program'])),
                    'model_sha256':nf.digest(nf.canonical(job['model'])),'formats':{}}
            for extension in ('tiff','gif'):
                image=output/(name+'.'+extension)
                image.write_bytes((source/('processing.'+extension)).read_bytes())
                result=nf.execute_carrier(image,work/(name+'-'+extension))['result']
                if result['active_cells']!=expected[name]:raise nf.Rejected('Variant behavior mismatch')
                continuation=nf.execute_carrier(work/(name+'-'+extension)/('processing.'+extension),work/(name+'-'+extension+'-refresh'))['result']
                record['formats'][extension]={'image_sha256':nf.digest(image.read_bytes()),'active_cells':result['active_cells'],
                    'final_tick':result['final_tick'],'refreshed_tick':continuation['final_tick'],
                    'refreshed_active_cells':continuation['active_cells'],'scalar_reference_pass':result['scalar_reference_all_ticks_pass']}
            report['cases'][name]=record
    report['changed_image_changes_computation']=True
    nf.write_json(output/'behavior_evidence.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('examples'))
    print(prepare(parser.parse_args().output))
