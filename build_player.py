# SPDX-License-Identifier: GPL-3.0-only
"""Build the generic VM, put its compiled bytes in an authoring payload, pin host trust."""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent

def build(dotnet):
    environment=os.environ.copy()
    environment.update(DOTNET_CLI_TELEMETRY_OPTOUT='1',DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_GENERATE_ASPNET_CERTIFICATE='false')
    environment.setdefault('DOTNET_CLI_HOME',str(ROOT/'runs/dotnet-home'))
    environment['APPDATA']=str(ROOT/'runs/dotnet-home/appdata')
    environment['NUGET_PACKAGES']=str(ROOT/'runs/nuget-packages')
    Path(environment['APPDATA']).mkdir(parents=True,exist_ok=True)
    for project in ('NeuralRuntime','NeuralPlayer'):
        subprocess.run([dotnet,'build',str(ROOT/'dotnet'/project/(project+'.csproj')),'-c','Release','--nologo',
                        '-p:RestoreConfigFile='+str(ROOT/'dotnet/NuGet.Config')],check=True,env=environment)
        if project=='NeuralRuntime':
            data=(ROOT/'dotnet/NeuralRuntime/bin/Release/net10.0/NeuralRuntime.dll').read_bytes()
            if not 1<=len(data)<=131072:raise ValueError('Runtime expansion limit')
            sha=hashlib.sha256(data).hexdigest()
            payload={'format':'dotnet-il/gzip-base64','sha256':sha,'data':base64.b64encode(gzip.compress(data,mtime=0)).decode('ascii')}
            if len(payload['data'])>20000:raise ValueError('Encoded runtime limit')
            (ROOT/'runtime-payload.json').write_text(json.dumps(payload,sort_keys=True,separators=(',',':'))+'\n',encoding='ascii')
            (ROOT/'dotnet/NeuralPlayer/TrustedRuntime.cs').write_text('// SPDX-License-Identifier: GPL-3.0-only\nnamespace NeuralPlayer;\ninternal static class TrustedRuntime { public const string Sha256="'+sha+'"; }\n',encoding='ascii')
            print(json.dumps({'runtime_bytes':len(data),'encoded_bytes':len(payload['data']),'sha256':sha}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dotnet',default=shutil.which('dotnet'));args=parser.parse_args()
    if not args.dotnet:parser.error('Install .NET 10 SDK 10.0.401 or specify --dotnet')
    build(args.dotnet)
