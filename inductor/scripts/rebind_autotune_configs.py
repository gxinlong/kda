"""Replay baseline configurations for generated modules changed only by kernel name.

This copies no graph, generated source, or binary cache. Full module ASTs (including
decorator precision/device/config metadata) must match after replacing only the
single Triton function's name and its references. Inductor still checks configs_hash.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(source):
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('triton_')]
    if len(functions)!=1:return None
    name=functions[0].name
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name==name:node.name='KDA_RENAMED_KERNEL'
            return self.generic_visit(node)
        def visit_Name(self,node):
            if node.id==name:node.id='KDA_RENAMED_KERNEL'
            return node
        def visit_Constant(self,node):
            if node.value==name:node.value='KDA_RENAMED_KERNEL'
            return node
    normalized=ast.dump(Rename().visit(tree),include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest(),name


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-manifest',type=Path,required=True)
    p.add_argument('--candidate-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();saved=json.loads(args.baseline_manifest.read_text());index={}
    for record in saved['records']:
        assert digest(record['source_path'])==record['source_sha256']
        assert digest(record['config_path'])==record['config_sha256']
        key=fingerprint(Path(record['source_path']).read_text())
        if key is None:continue
        fp,name=key
        if fp in index and index[fp][0]['config']!=record['config']:
            raise RuntimeError('Ambiguous baseline configurations for an identical module')
        index[fp]=(record,name)
    output=copy.deepcopy(saved);existing={r['source_relative'] for r in output['records']};mapped=[]
    cache=args.candidate_run/'cache'
    for path in sorted(cache.rglob('*.py')):
        relative=str(path.relative_to(cache))
        if relative in existing:continue
        source=path.read_text()
        if '\ndef triton_' not in source:continue
        key=fingerprint(source)
        if key is None or key[0] not in index:continue
        old,old_name=index[key[0]]
        record=copy.deepcopy(old)
        record.update(source_path=str(path.resolve()),source_relative=relative,source_sha256=digest(path))
        record['rebind_evidence']={'baseline_source_path':old['source_path'],
            'baseline_source_sha256':old['source_sha256'],'baseline_name':old_name,
            'candidate_name':key[1],'normalized_full_module_ast_sha256':key[0]}
        output['records'].append(record);existing.add(relative);mapped.append(record['rebind_evidence'])
    output['renamed_source_replay']={'baseline_manifest_sha256':digest(args.baseline_manifest),
        'candidate_run':str(args.candidate_run.resolve()),'mapped':mapped,
        'policy':'Only single kernel name/reference normalization in the complete module AST; all other code and metadata must match. Normal Inductor configs_hash validation remains enabled.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'baseline_records':len(saved['records']),
                      'renamed_records':len(mapped),'mapping':mapped},indent=2))


if __name__=='__main__':main()
