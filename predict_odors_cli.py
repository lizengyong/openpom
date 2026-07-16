import contextlib
import os

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import deepchem as dc  # must come BEFORE dgl on NPU (C-ext init order)
    import pandas as pd
    import numpy as np
    import torch
    import argparse
    import sys
    import time
    import dgl
    from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
    from openpom.models.mpnn_pom import MPNNPOMModel

TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]


def auto_select_device():
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return "npu:0"
    except (ImportError, AttributeError):
        pass
    return "cpu"


@contextlib.contextmanager
def suppress_all():
    with open(os.devnull, 'w') as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

def load_models(n_models=10, model_dir_prefix="./models/ensemble_models/experiments_", checkpoint="checkpoint2.pt", device_name=None):
    if device_name is None:
        device_name = auto_select_device()
    models = []
    for i in range(n_models):
        model = MPNNPOMModel(
            n_tasks=len(TASKS),
            batch_size=128,
            class_imbalance_ratio=None,
            loss_aggr_type='sum',
            node_out_feats=100,
            edge_hidden_feats=75,
            edge_out_feats=100,
            num_step_message_passing=5,
            mpnn_residual=True,
            message_aggregator_type='sum',
            mode='classification',
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1,
            readout_type='set2set',
            num_step_set2set=3,
            num_layer_set2set=2,
            ffn_hidden_list=[392, 392],
            ffn_embeddings=256,
            ffn_activation='relu',
            ffn_dropout_p=0.12,
            ffn_dropout_at_input_no_act=False,
            weight_decay=1e-5,
            self_loop=False,
            optimizer_name='adam',
            log_frequency=32,
            model_dir=f'{model_dir_prefix}{i+1}',
            device_name=device_name
        )
        model.restore(f"{model_dir_prefix}{i+1}/{checkpoint}")
        if device_name:
            model.model.to(torch.device(device_name))
        models.append(model)
    return models


def predict_odors(models_list, smiles):
    t_feat = time.perf_counter()
    featurizer = GraphFeaturizer()
    featurized_data = featurizer.featurize(smiles)
    t_feat = time.perf_counter() - t_feat

    preds = []
    times = []
    for i, model in enumerate(models_list):
        t_infer = time.perf_counter()
        prediction = model.predict(dc.data.NumpyDataset(featurized_data))
        elapsed = time.perf_counter() - t_infer
        times.append(elapsed)
        preds.append(prediction)
    preds_arr = np.asarray(preds)
    ensemble_preds = np.mean(preds_arr, axis=0)

    times = np.array(times)
    print(f"[Timing] featurize={t_feat:.3f}s  "
          f"inference: total={times.sum():.3f}s  "
          f"avg={times.mean():.3f}s  "
          f"min={times.min():.3f}s  max={times.max():.3f}s",
          file=sys.stderr)
    return ensemble_preds


def _verify_npu():
    device = torch.device(auto_select_device())
    print(f"[Verify] device = {device}")
    import warnings
    warnings.filterwarnings("ignore")

    from openpom.feat.graph_featurizer import GraphFeaturizer
    from openpom.models.mpnn_pom import MPNNPOM

    print("  [1/4] Featurizing SMILES ...", end=" ", flush=True)
    feat = GraphFeaturizer()
    graphs = feat.featurize(["CC", "C"])
    print("OK")

    print("  [2/4] Building DGL graph (CPU) ...", end=" ", flush=True)
    torch.manual_seed(0)
    dgl_graphs = [g.to_dgl_graph() for g in graphs]
    g_cpu = dgl.batch(dgl_graphs)
    print("OK")

    print("  [3/4] Creating model (CPU) ...", end=" ", flush=True)
    model = MPNNPOM(
        n_tasks=3, mode='classification',
        number_atom_features=134, number_bond_features=6,
        n_classes=1, ffn_embeddings=2,
    )
    print("OK")

    print(f"  [4/4] Moving to {device} and forward ...", end=" ", flush=True)
    g = g_cpu.to(device)
    model = model.to(device)
    with torch.no_grad():
        output = model(g)
    print("OK")

    proba, logits, embeddings = output
    proba_np = proba.detach().cpu().numpy()
    expected = np.asarray([[0.2934, 0.5467, 0.3940],
                           [0.4143, 0.6249, 0.3807]])
    ok = np.allclose(proba_np, expected, atol=0.001)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] proba match: got={proba_np.ravel()[:6]}  expect={expected.ravel()[:6]}")
    if not ok:
        sys.exit(1)
    print("[Verify] NPU forward test PASSED")


def main():
    parser = argparse.ArgumentParser(description="Predict odor profile from SMILES string")
    parser.add_argument("smiles", nargs="?", help="SMILES string to predict")
    parser.add_argument("-n", "--top-n", type=int, default=10, help="Number of top odors to display (default: 10)")
    parser.add_argument("--n-models", type=int, default=10, help="Number of ensemble models (default: 10)")
    parser.add_argument("--model-dir", default="./models/ensemble_models/experiments_", help="Model directory prefix")
    parser.add_argument("--checkpoint", default="checkpoint2.pt", help="Checkpoint filename")
    parser.add_argument("--device", default=None, help="Device: cuda, npu:0, cpu (default: auto-detect)")
    parser.add_argument("--show-all", action="store_true", help="Show all 138 odor predictions")
    parser.add_argument("--threshold", type=float, default=0.0, help="Only show odors with prediction >= threshold")
    parser.add_argument("-i", "--input-file", help="File containing SMILES strings (one per line)")
    parser.add_argument("--verify", action="store_true", help="Run NPU forward regression test (no checkpoint needed)")
    args = parser.parse_args()

    if args.verify:
        _verify_npu()
        return

    if not args.smiles and not args.input_file:
        parser.print_help()
        sys.exit(1)

    if args.input_file:
        with open(args.input_file) as f:
            all_smiles = [line.strip() for line in f if line.strip()]
    else:
        all_smiles = [args.smiles]

    with suppress_all():
        models_list = load_models(n_models=args.n_models, model_dir_prefix=args.model_dir,
                                  checkpoint=args.checkpoint, device_name=args.device)

    top_n = args.top_n if not args.show_all else len(TASKS)

    for smiles in all_smiles:
        predictions = predict_odors(models_list, [smiles])
        preds_arr = predictions.squeeze()

        pred_df = pd.DataFrame({"odor": TASKS, "prediction": preds_arr})
        pred_df = pred_df.sort_values("prediction", ascending=False)

        if args.threshold > 0.0:
            pred_df = pred_df[pred_df["prediction"] >= args.threshold]

        pred_df = pred_df.head(top_n)

        print(f"{'Odor':<20} {'Prediction':<12}")
        print("-" * 32)
        for _, row in pred_df.iterrows():
            print(f"{row['odor']:<20} {row['prediction']:<12.5f}")


if __name__ == "__main__":
    main()

