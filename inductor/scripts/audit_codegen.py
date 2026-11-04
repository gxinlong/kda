"""Audit embedded Triton code and launch helpers in immutable AOT packages.

Supports the inspected C++ wrapper format. Unsupported or incomplete extraction
fails; this is not a semantic proof for arbitrary wrappers or external kernels.
"""
import argparse
import ast
from collections import defaultdict
import copy
import json
from pathlib import Path
import re
import textwrap
import zipfile

from compile_aot import file_hash


class Normalize(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name

    def visit_FunctionDef(self, node):
        if node.name == self.name:
            node.name = 'KERNEL'
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == self.name:
            node.id = 'KERNEL'
        return node

    def visit_Constant(self, node):
        if node.value == self.name:
            node.value = 'KERNEL'
        return node

    def visit_keyword(self, node):
        # Only source-location/cache identity fields are ignored. Arithmetic,
        # backend options, signatures, constexprs, and configs remain visible.
        if node.arg == 'filename':
            node.value = ast.Constant('SOURCE_PATH')
        if node.arg == 'inductor_meta' and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value == 'backend_hash':
                    if not isinstance(value, ast.Constant):
                        raise ValueError('Unexpected backend_hash expression')
                    value.value = 'CACHE_ID'
        return self.generic_visit(node)


def inspect_wrapper(source):
    records = {}
    for name, raw in re.findall(r"async_compile\.triton\('([^']+)',\s*'''(.*?)''',", source, re.S):
        if name in records:
            raise ValueError('Duplicate kernel definition')
        tree = ast.parse(textwrap.dedent(raw).lstrip())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1 or functions[0].name != name:
            raise ValueError('Unexpected embedded function boundary')
        normalized = Normalize(name).visit(copy.deepcopy(tree))
        function, = [node for node in normalized.body if isinstance(node, ast.FunctionDef)]
        body = copy.deepcopy(function)
        body.decorator_list = []
        records[name] = {'module_ast': ast.dump(normalized, include_attributes=False),
                         'body_ast': ast.dump(body, include_attributes=False),
                         'decorators': [ast.unparse(node) for node in function.decorator_list]}
    pattern = r'static __attribute__\(\(noinline\)\) void call_(triton_\w+)\((.*?)launchKernel\((.*?)\);\s*\}'
    seen = set()
    for name, body, launch in re.findall(pattern, source, re.S):
        if name not in records or name in seen:
            raise ValueError('Unexpected/duplicate launch helper')
        seen.add(name)
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
        body = re.sub(r'loadKernel\("[^"]+"', 'loadKernel("CUBIN_PATH"', body)
        variables = {}
        def renumber(match):
            return variables.setdefault(match.group(), f'local_{len(variables)}')
        body = re.sub(r'\b(?:var|global_scratch_scratch|profile_scratch_scratch)_\d+\b', renumber, body)
        normalize = lambda value: re.sub(r'\s+', ' ', value.replace(name, 'KERNEL')).strip()
        records[name].update(launch_helper=normalize(body), launch_parameters=normalize(launch))
    all_launches = re.findall(r'launchKernel\(kernels_\.(triton_\w+),', source)
    if not records or set(records) != seen or len(all_launches) != len(records) or set(all_launches) != seen:
        raise ValueError('Incomplete wrapper extraction; inspect this compiler version manually')
    return records


def compare(before, after):
    unmatched = set(after)
    pairs, changes = [], []
    # Exact normalized matches first, with each candidate consumed only once.
    for name, record in before.items():
        matches = sorted(candidate for candidate in unmatched if after[candidate] == record)
        if matches:
            match = name if name in matches else matches[0]
            pairs.append({'baseline': name, 'candidate': match, 'changed': []})
            unmatched.remove(match)
        else:
            changes.append(name)
    for name in changes:
        record = before[name]
        matches = [candidate for candidate in unmatched if after[candidate]['body_ast'] == record['body_ast']]
        match = name if name in unmatched else (matches[0] if len(matches) == 1 else None)
        if match is None:
            pairs.append({'baseline': name, 'candidate': None, 'changed': ['removed_or_ambiguous'],
                          'possible_matches': sorted(matches)})
            continue
        unmatched.remove(match)
        pairs.append({'baseline': name, 'candidate': match,
                      'changed': [key for key in record if record[key] != after[match][key]]})
    return {'pairs': pairs, 'added_or_unmatched': sorted(unmatched)}


def package(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    info = manifest['package']
    if file_hash(info['path']) != info['sha256']:
        raise ValueError('Immutable package changed')
    with zipfile.ZipFile(info['path']) as archive:
        wrappers = [name for name in archive.namelist() if name.endswith('.wrapper.cpp')]
        if len(wrappers) != 1:
            raise ValueError('Expected exactly one C++ wrapper; choose a model explicitly')
        source = archive.read(wrappers[0]).decode()
    return info, source, inspect_wrapper(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before, before_source, before_records = package(args.baseline_run)
    after, after_source, after_records = package(args.candidate_run)
    result = {'baseline': before, 'candidate': after, **compare(before_records, after_records),
              'records': {'baseline': before_records, 'candidate': after_records},
              'scope': 'Triton module AST and actual C++ launch helper only. Kernel names, source/cache paths, backend_hash and local variable suffixes are normalized. No proof of full wrapper, external-library behavior, or binary equivalence.'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'baseline.wrapper.cpp').write_text(before_source)
    (args.output / 'candidate.wrapper.cpp').write_text(after_source)
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'changed_pairs': sum(bool(row['changed']) for row in result['pairs']),
                      'added_or_unmatched': result['added_or_unmatched']}))


if __name__ == '__main__':
    main()
