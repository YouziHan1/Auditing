import time
from pathlib import Path
import torch

import datasets.SETUP as SETUP
import datasets.utils as dms_utils
import datasets.model as dms_model
import utils
import train_scheduler as tsch


def run(args):
    start = time.time()
    SETUP.setup_seed(args.seed)
    device = SETUP.get_device()

    # load dataset
    dataset, split = dms_utils.get_raw_dataset(args.dataset)
    graph = dataset[0]
    args.num_classes = dataset.num_classes

    # ensure args has common defaults expected by trainer
    if not hasattr(args, 'num_neighbors'):
        args.num_neighbors = 1
    if not hasattr(args, 'num_not_neighbors'):
        args.num_not_neighbors = 1
    if not hasattr(args, 'num_neighbors_test'):
        args.num_neighbors_test = 1
    if not hasattr(args, 'worker_num'):
        args.worker_num = 4
    if not hasattr(args, 'priv_epsilon'):
        args.priv_epsilon = 8.8
    if not hasattr(args, 'log_dir'):
        args.log_dir = 'logs'
    if not hasattr(args, 'C'):
        args.C = 1.0
    if not hasattr(args, 'graph_setting'):
        args.graph_setting = 'transductive'

    # build model (use G_net as default)
    model = dms_model.G_net(K=args.K, feat_dim=graph.x.shape[1], num_classes=args.num_classes, hidden_channels=args.hidden_channels)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare train/test loaders using utils helper if available
    try:
        train_loader, val_loader, test_loader, dataset_obj, x = dms_utils.form_loaders(args)
    except Exception:
        # fallback: construct simple TensorDataset on node features and masks
        train_mask = split.train_mask
        test_mask = split.test_mask
        train_dataset = torch.utils.data.TensorDataset(graph.x[train_mask], graph.y[train_mask])
        test_dataset = torch.utils.data.TensorDataset(graph.x[test_mask], graph.y[test_mask])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.expected_batchsize, shuffle=True, num_workers=4)
        val_loader = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.expected_batchsize, shuffle=False, num_workers=4)

    trainer = tsch.trainer(
        model=model,
        optimizer=optimizer,
        loaders=[train_loader, val_loader, test_loader],
        device=device,
        criterion=dms_model.criterion,
        args=args,
    )

    # run training (will update model in-place). Do not save.
    trainer.run()

    # after training finished, enter interactive prediction loop
    model.eval()
    feat_dim = graph.x.shape[1]

    # print feature-dimension and provide one training example as reference
    print(f'Feature dimension (expected input length) = {feat_dim}')
    try:
        train_mask = split.train_mask
        train_idxs = train_mask.nonzero(as_tuple=True)[0]
        if train_idxs.numel() > 0:
            example_idx = int(train_idxs[0])
            example_feat = graph.x[example_idx].tolist()
            example_label = int(graph.y[example_idx].item()) if hasattr(graph.y[example_idx], 'item') else int(graph.y[example_idx])
            if feat_dim <= 50:
                print(f'Example training node index: {example_idx}')
                print('Example feature vector (full):')
                print(example_feat)
            else:
                print(f'Example training node index: {example_idx}')
                print('Example feature vector (first 20 dims):')
                print(example_feat[:20], '...')
            print(f'Example label: {example_label}')
            # show example input string for interactive mode
            example_input_str = ','.join([str(x) for x in (example_feat if feat_dim <= 50 else example_feat[:feat_dim])])
            print('To predict using raw features, input comma-separated values of length', feat_dim)
            print('Example raw-feature input (truncated if long):')
            print(example_input_str if len(example_input_str) < 400 else example_input_str[:400] + '...')
    except Exception as _:
        # ignore if split or graph fields not present
        pass

    def predict_from_tensor(x_tensor):
        x_tensor = x_tensor.to(device)
        with torch.no_grad():
            out = model(x_tensor)
            pred = out.argmax(dim=1).item()
        return pred

    # if node_index or raw_feature provided via args, do one-time prediction first
    if args.node_index is not None:
        idx = int(args.node_index)
        if idx < 0 or idx >= graph.x.shape[0]:
            print(f'Node index {idx} out of range [0, {graph.x.shape[0]-1}]')
        else:
            x = graph.x[idx].unsqueeze(0)
            pred = predict_from_tensor(x)
            print(f'==> Predicted class for node {idx}: {pred}')

    if args.raw_feature is not None:
        try:
            vals = [float(x) for x in args.raw_feature.split(',')]
            if len(vals) != feat_dim:
                print(f'Provided feature length {len(vals)} != expected {feat_dim}')
            else:
                x = torch.tensor(vals, dtype=graph.x.dtype).unsqueeze(0)
                pred = predict_from_tensor(x)
                print(f'==> Predicted class for provided feature: {pred}')
        except Exception as e:
            print('Error parsing --raw_feature:', e)

    # interactive loop
    print('\nInteractive mode: enter a node index, or a comma-separated raw feature vector, or "q" to quit.')
    while True:
        try:
            s = input('predict> ').strip()
        except EOFError:
            print('\nexiting')
            break
        if s.lower() in ('q', 'quit', 'exit') or s == '':
            print('bye')
            break

        # try node index first
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            idx = int(s)
            if idx < 0 or idx >= graph.x.shape[0]:
                print(f'Node index {idx} out of range [0, {graph.x.shape[0]-1}]')
                continue
            x = graph.x[idx].unsqueeze(0)
            pred = predict_from_tensor(x)
            print(f'Predicted class for node {idx}: {pred}')
            continue

        # otherwise try parse as raw feature
        parts = [p.strip() for p in s.split(',') if p.strip()!='']
        if len(parts) > 0:
            try:
                vals = [float(x) for x in parts]
            except Exception as e:
                print('Could not parse input as numbers or node index:', e)
                continue
            if len(vals) != feat_dim:
                print(f'Feature length {len(vals)} does not match expected {feat_dim}')
                continue
            x = torch.tensor(vals, dtype=graph.x.dtype).unsqueeze(0)
            pred = predict_from_tensor(x)
            print(f'Predicted class for provided feature: {pred}')
            continue

        print('Unrecognized input. Enter a node index (e.g. 0) or comma-separated features, or q to quit.')

    print(f'Total time: {time.time()-start:.2f}s')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train on dataset and predict one example without saving')
    parser.add_argument('--dataset', type=str, default='Amazon_Computers', required=True)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--expected_batchsize', type=int, default=128)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--node_index', type=int, default=None, help='index of test node to predict')
    parser.add_argument('--raw_feature', type=str, default=None, help='comma separated raw feature vector to predict')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
