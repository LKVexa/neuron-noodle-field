# SPDX-License-Identifier: GPL-3.0-only
"""Execute and refresh image-carried neural operator graphs, models and state."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import uuid
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import graph_vm

VERSION = "0.4.0"
MAGIC = b"NNFIELD2"
WIDTH, HEIGHT, Y0, CELL = 960, 720, 512, 1
MAX_PAYLOAD = 16384
MAX_FILE = 32 * 1024 * 1024
MAX_FRAMES = 32
MAX_TOTAL_PIXELS = WIDTH * HEIGHT * MAX_FRAMES
DEFAULT_PREPARATION = {"schema": "tgc-job/1", "kind": "neural-field",
               "params": {"size": 32, "steps": 12, "seed": 7, "epochs": 2400}}


class Rejected(ValueError):
    """An input is outside the supported, bounded demonstration profile."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True,
                                    allow_nan=False) + "\n", encoding="utf-8")


def reject_constant(value):
    raise Rejected("Nonfinite JSON value: " + value)


def bounded_bytes(path, limit):
    """Read at most limit+1 bytes, without trusting an earlier file size check."""
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise Rejected("File resource limit exceeded")
    return data


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise Rejected("Duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=pairs)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise Rejected("Malformed, duplicate, or nonfinite JSON") from exc


def validate_job(job):
    fields={"schema","kind","program","model","state","tick","steps"}
    if type(job) is not dict or set(job)!=fields or job['schema']!='nnf-image/1' or job['kind']!='neural-graph':
        raise Rejected("Image must carry a program, model, binary state, tick and step count")
    if type(job['tick']) is not int or type(job['steps']) is not int or not 0<=job['tick']<=1_000_000 or not 1<=job['steps']<=24 or job['tick']+job['steps']>1_000_000:
        raise Rejected("Image tick/step resource limit exceeded")
    state=job['state']
    if type(state) is not list or not 8<=len(state)<=64 or any(type(row) is not list or len(row)!=len(state) for row in state):
        raise Rejected("Image state must be a square of size 8..64")
    if any(type(value) is not int or value not in (0,1) for row in state for value in row):
        raise Rejected("Image state values must be integer bits")
    try:
        graph_vm.validate_program(job['program'])
        graph_vm.validate_model(job['model'])
        # Type/shape admission is separate from accepting any resulting state.
        graph_vm.execute(job['program'],job['model'],np.asarray(state,dtype=np.uint8))
    except graph_vm.GraphError as exc:
        raise Rejected(str(exc)) from exc
    if len(canonical(job))>MAX_PAYLOAD: raise Rejected("Image program/model/state payload limit exceeded")
    return job


def default_program():
    return {"schema":"nnf-graph/1","code":[["STATE",0],
        ["GATHER",1,0,[[0,0],[-1,0],[1,0],[0,-1],[0,1]]],
        ["PARAM",2,"weights"],["DOT",3,1,2],["PARAM",4,"bias"],
        ["ADD",5,3,4],["SIGMOID",6,5],["CONST",7,0.5],["GE",8,6,7],["RETURN",8,6]]}


def make_image_job(state,program,model,*,tick=0,steps=1):
    validate_state(state)
    job={"schema":"nnf-image/1","kind":"neural-graph","program":program,"model":model,
         "state":state.tolist(),"tick":tick,"steps":steps}
    # Snapshot caller-owned graph/model containers through the serialized domain.
    return validate_job(strict_json(canonical(job)))


def prepare_image_job(*,size=32,steps=12,seed=7,epochs=2400):
    if type(size) is not int or not 8<=size<=64: raise Rejected("Preparation size must be 8..64")
    weights,bias,_,_,_=train_local_or(seed,epochs)
    check_model_truth_table(weights,bias)
    state=np.zeros((size,size),dtype=np.uint8); state[size//2,size//2]=1
    return make_image_job(state,default_program(),{"weights":weights.tolist(),"bias":bias},steps=steps)


def font(size=24):
    return ImageFont.load_default(size=size)


def frame(title, subtitle=""):
    im = Image.new("RGB", (WIDTH, HEIGHT), "#101a2b")
    draw = ImageDraw.Draw(im)
    draw.text((32, 24), "NEURON NOODLE FIELD", font=font(18), fill="#7bdad0")
    draw.text((32, 60), title, font=font(31), fill="white")
    draw.text((32, 103), subtitle[:90], font=font(17), fill="#aec0d7")
    draw.text((32, 464), "Pixels carry program + model + state | generic CPU tensor interpreter",
              font=font(16), fill="#aec0d7")
    return im


def encode_payload(im, job):
    validate_job(job)
    if not isinstance(im, Image.Image) or im.size != (WIDTH, HEIGHT):
        raise Rejected("Carrier dimensions do not match the profile")
    raw = canonical(job)
    if len(raw) > MAX_PAYLOAD:
        raise Rejected("Payload resource limit exceeded")
    packet = MAGIC + struct.pack(">I", len(raw)) + hashlib.sha256(raw).digest() + raw
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    cols = WIDTH // CELL
    rows = (len(bits) + cols - 1) // cols
    if Y0 + rows * CELL > HEIGHT:
        raise Rejected("Image payload capacity exceeded")
    plane = np.full((rows, cols), 255, dtype=np.uint8)
    plane.flat[:len(bits)] = bits * 255
    tile = Image.fromarray(np.repeat(np.repeat(plane, CELL, 0), CELL, 1)).convert("RGB")
    result = im.convert("RGB")
    result.paste(tile, (0, Y0))
    return result


def decode_payload(im):
    if im.size != (WIDTH, HEIGHT):
        raise Rejected("Carrier dimensions do not match the profile")
    rgb = np.asarray(im.convert("RGB"))
    def take_bytes(n):
        count = n * 8
        rows = (count + WIDTH // CELL - 1) // (WIDTH // CELL)
        if Y0 + rows * CELL > HEIGHT:
            raise Rejected("Truncated carrier")
        block = rgb[Y0:Y0 + rows * CELL].reshape(rows, CELL, WIDTH // CELL, CELL, 3)
        cells = block.transpose(0, 2, 1, 3, 4).reshape(-1, CELL * CELL * 3)[:count]
        if not np.all((cells == 0) | (cells == 255)) or not np.all(cells == cells[:, :1]):
            raise Rejected("Carrier cells are damaged or resampled")
        return np.packbits((cells[:, 0] // 255).astype(np.uint8)).tobytes()
    header = take_bytes(44)
    if header[:8] != MAGIC:
        raise Rejected("Not a supported raster carrier")
    length = struct.unpack(">I", header[8:12])[0]
    if not 1 <= length <= MAX_PAYLOAD:
        raise Rejected("Payload resource limit exceeded")
    raw = take_bytes(44 + length)[44:]
    if hashlib.sha256(raw).digest() != header[12:44]:
        raise Rejected("Payload checksum mismatch")
    job = strict_json(raw)
    try:
        if canonical(job) != raw:
            raise Rejected("Noncanonical payload")
    except (ValueError, RecursionError) as exc:
        raise Rejected("Invalid canonical payload") from exc
    return validate_job(job)


def read_carrier(path):
    raw = bounded_bytes(path, MAX_FILE)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in ("TIFF", "GIF", "PNG"):
                    raise Rejected("Unsupported carrier format")
                count = getattr(im, "n_frames", 1)
                if not 1 <= count <= MAX_FRAMES:
                    raise Rejected("Animation frame limit exceeded")
                first, expected, total = None, None, 0
                for index in range(count):
                    im.seek(index)
                    if im.size != (WIDTH, HEIGHT):
                        raise Rejected("Carrier dimensions do not match the profile")
                    total += im.width * im.height
                    if total > MAX_TOTAL_PIXELS:
                        raise Rejected("Decoded pixel limit exceeded")
                    job = decode_payload(im)
                    if expected is None:
                        expected, first = job, im.convert("RGB")
                    elif job != expected:
                        raise Rejected("Animation frames contain different programs")
                return expected, first
    except (UnidentifiedImageError, OSError, EOFError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise Rejected("Malformed or excessive image") from exc


def save_carriers(folder, frames, job):
    validate_job(job)
    if not isinstance(frames, (list, tuple)) or not 1 <= len(frames) <= MAX_FRAMES:
        raise Rejected("Carrier frame limit exceeded")
    encoded = [encode_payload(im, job) for im in frames]
    palette = [value for r in range(6) for g in range(6) for b in range(6)
               for value in (r * 51, g * 51, b * 51)]
    palette += [0] * (768 - len(palette))
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(palette)
    gifs = [im.quantize(palette=palette_image, dither=Image.Dither.NONE) for im in encoded]
    folder = Path(folder)
    encoded[0].save(folder / "processing.tiff", save_all=True,
                    append_images=encoded[1:], compression="tiff_deflate")
    gifs[0].save(folder / "processing.gif", save_all=True, append_images=gifs[1:],
                 duration=420, loop=0, optimize=False, disposal=2)
    checks = {}
    for name in ("processing.tiff", "processing.gif"):
        recovered, _ = read_carrier(folder / name)
        if recovered != job:
            raise Rejected("Carrier roundtrip mismatch")
        checks[name] = {"sha256": digest(bounded_bytes(folder / name, MAX_FILE)),
                        "all_frame_payload_roundtrip": True}
    return checks


def validate_state(state):
    if not isinstance(state, np.ndarray) or state.dtype != np.uint8 or state.ndim != 2:
        raise Rejected("State must be a two-dimensional uint8 array")
    if state.shape[0] != state.shape[1] or not 8 <= state.shape[0] <= 64:
        raise Rejected("State must be a square of 8..64 cells per side")
    if not np.all((state == 0) | (state == 1)):
        raise Rejected("State must contain binary values only")
    return state


def validate_model(weights, bias):
    if not isinstance(weights, np.ndarray) or weights.shape != (5,) or weights.dtype.kind != "f":
        raise Rejected("Model requires five floating-point weights")
    if type(bias) not in (float, int) or abs(bias) > 1e6 or not math.isfinite(bias):
        raise Rejected("Model bias must be finite")
    if not np.isfinite(weights).all() or np.max(np.abs(weights)) > 1e6 or abs(bias) > 1e6:
        raise Rejected("Nonfinite or excessive model parameters")
    return weights, float(bias)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def local_truth_table():
    inputs = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(np.float64)
    targets = np.any(inputs, axis=1)
    return inputs, targets


def train_local_or(seed, epochs):
    if type(seed) is not int or type(epochs) is not int or not 0 <= seed <= 65535 or not 200 <= epochs <= 4000:
        raise Rejected("Training resource limit exceeded")
    inputs, targets = local_truth_table()
    weights = np.random.default_rng(seed).normal(0, .01, 5)
    bias = 0.0
    for _ in range(epochs):
        error = sigmoid(inputs @ weights + bias) - targets
        weights -= .8 * (inputs.T @ error) / 32
        bias -= .8 * float(error.mean())
    validate_model(weights, bias)
    return weights, bias, inputs, targets, sigmoid(inputs @ weights + bias)


def pixel160_state(probabilities, state, tick):
    validate_state(state)
    if type(tick) is not int or not 0 <= tick <= 1_000_000:
        raise Rejected("Tick outside the local profile")
    if not isinstance(probabilities, np.ndarray) or probabilities.shape != state.shape or probabilities.dtype.kind not in "fiu":
        raise Rejected("Probability shape mismatch")
    if not np.isfinite(probabilities).all() or not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise Rejected("Probability range exceeded")
    lanes = np.zeros((state.size, 5), dtype=np.uint32)
    lanes[:, 0] = np.asarray(probabilities, dtype=np.float32).ravel().view(np.uint32)
    lanes[:, 1], lanes[:, 2], lanes[:, 3] = state.ravel(), tick, 1
    return lanes.astype(">u4").tobytes()


def check_model_truth_table(weights, bias):
    validate_model(weights, bias)
    inputs, targets = local_truth_table()
    if not np.array_equal(sigmoid(inputs @ weights + bias) >= .5, targets):
        raise Rejected("Model does not satisfy all 32 local truth-table cases")


def save_checkpoint(folder, state, weights, bias, tick, program):
    validate_state(state)
    graph_vm.validate_model({'weights':weights.tolist(),'bias':bias})
    program=graph_vm.validate_program(program)
    if type(tick) is not int or not 0 <= tick <= 1_000_000:
        raise Rejected("Checkpoint tick outside the profile")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    Image.fromarray(state * 255).save(folder / "state.tiff", compression="tiff_deflate")
    write_json(folder / "model.json", {"weights": weights.tolist(), "bias": float(bias)})
    write_json(folder / "program.json",program)
    manifest = {"schema": "nnf-checkpoint/2", "version": VERSION, "tick": tick,
        "shape": list(state.shape), "state_sha256": digest((folder / "state.tiff").read_bytes()),
        "model_sha256": digest((folder / "model.json").read_bytes()),
        "program_sha256": digest((folder / "program.json").read_bytes()),
        "binary_state_sha256": digest(state.tobytes())}
    write_json(folder / "checkpoint.json", manifest)
    return manifest


def load_checkpoint(folder):
    folder = Path(folder)
    manifest = strict_json(bounded_bytes(folder / "checkpoint.json", 8192))
    expected = {"schema", "version", "tick", "shape", "state_sha256", "model_sha256", "program_sha256", "binary_state_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != expected or manifest["schema"] != "nnf-checkpoint/2" or manifest["version"] != VERSION:
        raise Rejected("Unsupported checkpoint manifest")
    shape, tick = manifest["shape"], manifest["tick"]
    if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int for n in shape) or shape[0] != shape[1] or not 8 <= shape[0] <= 64:
        raise Rejected("Invalid checkpoint dimensions")
    if type(tick) is not int or not 0 <= tick <= 1_000_000:
        raise Rejected("Invalid checkpoint tick")
    for key in ("state_sha256", "model_sha256", "program_sha256", "binary_state_sha256"):
        value = manifest[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise Rejected("Invalid checkpoint digest")
    model_raw = bounded_bytes(folder / "model.json", 4096)
    program_raw=bounded_bytes(folder / "program.json",8192)
    state_raw = bounded_bytes(folder / "state.tiff", 1024 * 1024)
    if digest(model_raw) != manifest["model_sha256"] or digest(state_raw) != manifest["state_sha256"] or digest(program_raw)!=manifest['program_sha256']:
        raise Rejected("Checkpoint integrity mismatch")
    model = strict_json(model_raw)
    if not isinstance(model, dict) or set(model) != {"weights", "bias"} or not isinstance(model["weights"], list) or not 1<=len(model["weights"])<=9:
        raise Rejected("Invalid checkpoint model")
    if any(type(value) not in (float, int) or abs(value) > 1e6 for value in model["weights"]):
        raise Rejected("Invalid weight type")
    weights, bias = np.asarray(model["weights"], dtype=np.float64), model["bias"]
    try:
        graph_vm.validate_model(model)
        program=graph_vm.validate_program(strict_json(program_raw))
    except graph_vm.GraphError as exc: raise Rejected(str(exc)) from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(state_raw)) as im:
                if im.format != "TIFF" or getattr(im, "n_frames", 1) != 1 or list(im.size) != shape or im.mode != "L":
                    raise Rejected("Invalid checkpoint TIFF profile")
                pixels = np.asarray(im).copy()
    except (OSError, EOFError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise Rejected("Invalid checkpoint image") from exc
    if not np.all((pixels == 0) | (pixels == 255)):
        raise Rejected("Checkpoint pixels must be exactly 0 or 255")
    state = (pixels // 255).astype(np.uint8)
    validate_state(state)
    if digest(state.tobytes()) != manifest["binary_state_sha256"]:
        raise Rejected("Checkpoint binary state mismatch")
    try: graph_vm.execute(program,model,state)
    except graph_vm.GraphError as exc: raise Rejected(str(exc)) from exc
    return state, weights, float(bias), tick, manifest, program


def _render_state(state, tick):
    image = frame("Image-carried neural computation", f"Generic operator graph | accepted tick {tick}")
    display = Image.fromarray(state * 255).resize((288, 288), Image.Resampling.NEAREST).convert("RGB")
    image.paste(display, (48, 154))
    draw = ImageDraw.Draw(image)
    for y, text in [(180, f"Active cells: {int(state.sum())}"), (230, "Program + weights in pixels"),
                    (280, "Independent scalar executor"), (325, "State retained on refresh")]:
        draw.text((382, y), text, font=font(22), fill="white")
    return image


def run_neural(job, folder):
    validate_job(job)
    folder = Path(folder)
    program,model=job['program'],job['model']
    weights,bias=np.asarray(model['weights'],dtype=np.float64),float(model['bias'])
    state=np.asarray(job['state'],dtype=np.uint8).copy(); n=len(state)
    states, frames, roots, probabilities = [state.copy()], [], [], state.astype(float)
    show_ticks = set(np.linspace(0, job["steps"], min(12, job["steps"] + 1), dtype=int))
    for step in range(job['steps']+1):
        tick=job['tick']+step
        roots.append(digest(b"tgc-pixel160/1\0" + pixel160_state(probabilities, state, tick)))
        if step in show_ticks:
            frames.append(_render_state(state, tick))
        if step < job["steps"]:
            try: state,probabilities=graph_vm.checked_step(program,model,state)
            except graph_vm.GraphError as exc: raise Rejected(str(exc)) from exc
            states.append(state.copy())
    midpoint = job['steps']//2
    save_checkpoint(folder / "checkpoint", states[midpoint], weights, bias, job['tick']+midpoint,program)
    restored,rw,rb,rtick,_,restored_program=load_checkpoint(folder/'checkpoint')
    restored_model={'weights':rw.tolist(),'bias':rb}
    for _ in range(rtick,job['tick']+job['steps']):
        restored,restored_probabilities=graph_vm.checked_step(restored_program,restored_model,restored)
    if not np.array_equal(restored, state) or not np.array_equal(restored_probabilities, probabilities):
        raise Rejected("Checkpoint continuation differs from uninterrupted execution")
    pages = [Image.fromarray(s * 255) for s in states]
    pages[0].save(folder / "field_states.tiff", save_all=True, append_images=pages[1:], compression="tiff_deflate")
    final_tick=job['tick']+job['steps']
    final = pixel160_state(probabilities, state, final_tick)
    (folder / "accepted_pixel160.bin").write_bytes(final)
    write_json(folder / "coordinates.json", {"profile": "tgc-neural-pixel160/1", "order": "row-major",
        "axes": ["t", "z", "y", "x", "c"], "shape": [1, 1, n, n, 1], "tick": final_tick, "field_root": roots[-1]})
    next_job=make_image_job(state,program,model,tick=final_tick,steps=1)
    return {"program_from_image":True,"model_from_image":True,"initial_state_from_image":True,
        "program_sha256":digest(canonical(program)),"model_sha256":digest(canonical(model)),
        "scalar_reference_all_ticks_pass":True,"checkpoint_continuation_pass":True,
        "continuation_steps":job['steps']-midpoint,"initial_tick":job['tick'],"final_tick":final_tick,
        "active_cells":int(state.sum()),"state_roots":roots,"accepted_state_count":len(states)},frames,next_job


def write_mssl_evidence(path, receipt):
    """Write the observed MSSL module/learn evidence profile, not kernel code."""
    module = "neuron.noodle.field.evidence"
    facts = {"status": "HOST_EXECUTION_RECORDED", "version": VERSION,
        "receipt_sha256": digest(canonical(receipt)), "result": receipt["result"],
        "execution_authority": "bounded_python_numpy_cpu", "native_mssl_computation": False,
        "physical_hologram": False, "release_qualification": False}
    js = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    rules = {"0001": "module(" + js(module) + ")"}
    for i, (key, value) in enumerate(facts.items(), 2):
        rules[f"{i:04d}"] = "learn(" + js(key) + ",literal(" + js(js(value)) + "))"
    fields = [("α", {"format":"0.3", "profile":"basic_mssl.executable.3", "record":"MSSL.Document", "source_language":"MSSL", "source_version":"0.3"}),
        ("ι", "mssl:neuron:noodle:field:evidence:"+VERSION), ("λ", module), ("ℛ", rules), ("π", []),
        ("δ", len(rules)), ("τ", 0), ("μ", {"learn."+key:value for key,value in facts.items()}),
        ("ρ", {"assertions":[], "emissions":[], "prints":[]}), ("χ", []),
        ("η", {"events":[], "state":facts}), ("ν", {})]
    body = ",\n".join(key + "≡" + js(value) for key,value in fields) + "\n"
    Path(path).write_text("MSSL{\n" + body + ",κ≡" + js("sha256:" + digest(body.encode("utf-8"))) + "\n}⇒⊤\n", encoding="utf-8")


def _new_output(output):
    output = Path(output).resolve()
    if output.exists():
        raise Rejected("Output already exists; choose a new folder")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _publish(staging, output):
    if output.exists():
        raise Rejected("Output appeared during execution; refusing to replace it")
    staging.rename(output)


@contextmanager
def _staging_output(parent, prefix):
    """Create a staging folder with normal inherited output permissions.

    Windows TemporaryDirectory creates an owner-only ACL. Renaming such a
    folder into a deliverable would retain that private ACL, hiding output from
    the desktop user or other tools. Ordinary mkdir inherits the parent ACL.
    """
    parent = Path(parent).resolve()
    staging = parent / (prefix + uuid.uuid4().hex)
    staging.mkdir(mode=0o777)
    try:
        yield staging
    finally:
        if staging.exists():
            if staging.parent != parent or not staging.name.startswith(prefix):
                raise Rejected("Unexpected staging cleanup target")
            shutil.rmtree(staging)


def execute_carrier(image_path, output):
    image_path = Path(image_path).resolve()
    raw = bounded_bytes(image_path, MAX_FILE)
    # Decode exactly the bytes retained in the receipt, even if the source changes.
    with tempfile.TemporaryDirectory(prefix="nnf-source-") as input_dir:
        stable_source = Path(input_dir) / "source.image"
        stable_source.write_bytes(raw)
        job, _ = read_carrier(stable_source)
    output = _new_output(output)
    with _staging_output(output.parent, ".nnf-") as staging:
        suffix = {b"GIF": ".gif", b"\x89PN": ".png"}.get(raw[:3], ".tiff")
        source_name = "input" + suffix
        (staging / source_name).write_bytes(raw)
        result, frames, next_job = run_neural(job, staging)
        carriers = save_carriers(staging, frames, next_job)
        receipt = {"tool": "Neuron Noodle Field", "version": VERSION,
            "claim": "pixels carry executable operator graph, model and state; generic CPU interpretation; no physical hologram",
            "source_path": source_name, "source_sha256": digest(raw),
            "job": job, "job_sha256": digest(canonical(job)), "result": result, "carriers": carriers,
            "output_job_sha256":digest(canonical(next_job)),"output_behavior":"refresh advances carried state by one step"}
        write_json(staging / "receipt.json", receipt)
        write_mssl_evidence(staging / "evidence.mssl", receipt)
        _publish(staging, output)
    return receipt


def demo(output, job=None):
    job = validate_job(prepare_image_job() if job is None else job)
    with tempfile.TemporaryDirectory(prefix="nnf-demo-") as temp:
        source = Path(temp) / "input.tiff"
        encode_payload(frame("The program and model are in the pixels", "Generic tensor instructions execute the state carried by this image"), job).save(source, compression="tiff_deflate")
        return execute_carrier(source, output)


def resume_checkpoint(checkpoint, steps, output):
    if type(steps) is not int or not 1 <= steps <= 24:
        raise Rejected("Resume steps must be 1..24")
    state,weights,bias,tick,manifest,program=load_checkpoint(checkpoint)
    job=make_image_job(state,program,{'weights':weights.tolist(),'bias':bias},tick=tick,steps=steps)
    with tempfile.TemporaryDirectory(prefix='nnf-checkpoint-source-') as temp:
        source=Path(temp)/'checkpoint_input.tiff'
        encode_payload(frame('Restored image program and state'),job).save(source,compression='tiff_deflate')
        return execute_carrier(source,output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    subs = parser.add_subparsers(dest="command")
    d = subs.add_parser("demo", help="train, validate, and create a replayable demo")
    d.add_argument("--output", type=Path, default=Path("runs/demo"))
    for name, default in DEFAULT_PREPARATION["params"].items():
        d.add_argument("--" + name, type=int, default=default)
    r = subs.add_parser("run", help="replay a neural TIFF/GIF/PNG carrier")
    r.add_argument("image", type=Path)
    r.add_argument("--output", type=Path, default=Path("runs/replayed"))
    inspect = subs.add_parser("inspect", help="validate every carrier frame and print its job")
    inspect.add_argument("image", type=Path)
    resume = subs.add_parser("resume", help="continue an integrity-checked checkpoint")
    resume.add_argument("checkpoint", type=Path)
    resume.add_argument("--steps", type=int, required=True)
    resume.add_argument("--output", type=Path, default=Path("runs/resumed"))
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "demo"):
            job = prepare_image_job(**{key:getattr(args,key,value) for key,value in DEFAULT_PREPARATION['params'].items()})
            result = demo(getattr(args, "output", Path("runs/demo")), job)
        elif args.command == "run": result = execute_carrier(args.image, args.output)
        elif args.command == "resume": result = resume_checkpoint(args.checkpoint, args.steps, args.output)
        else: result = read_carrier(args.image)[0]
    except (Rejected, OSError,graph_vm.GraphError) as exc:
        print("Rejected: " + str(exc), file=sys.stderr)
        return 2
    display=result if args.command=='inspect' else {'version':VERSION,'result':result['result'],
        'carriers':result['carriers'],'output_behavior':result['output_behavior']}
    print(json.dumps(display, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
