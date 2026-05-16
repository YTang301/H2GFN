import torch
import copy
import timeit
import warnings
import torch.optim as optim
from pre_MAED import pre_MAED,Similarity_graph

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MAED(MDA, interaction, dissim, mirsim, args):
    data = {
        'diss': dissim,
        'mis': mirsim,
        'disease_number': dissim.shape[0],
        'miRNA_number': mirsim.shape[0]
    }

    d_graph, m_graph = Similarity_graph(data, args)

    tasks = {
        'disease': {
            'graph': d_graph.to(device),
            'feature': torch.FloatTensor(dissim).to(device),
            'dim': data['disease_number']
        },
        'miRNA': {
            'graph': m_graph.to(device),
            'feature': torch.FloatTensor(mirsim).to(device),
            'dim': data['miRNA_number']
        }
    }

    new_X = {}
    start = timeit.default_timer()

    for item, config in tasks.items():
        entity_graph = config['graph']
        entity_feature = config['feature']
        model = pre_MAED(args, in_dim=config['dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_loss = float('inf')
        best_model = None
        start_alpha, end_alpha = 1,3
        for epoch in range(args.epochs):
            model.train()
            current_alpha = start_alpha + (end_alpha - start_alpha) * (epoch / args.epochs)
            loss, _, _ = model(entity_graph, entity_feature, alpha=current_alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = copy.deepcopy(model)

        best_model.eval()
        with torch.no_grad():
            new_X[item] = best_model.encoding_maed(entity_graph, entity_feature)

        print(f"Entity: {item} | Best Training Loss: {best_loss:.8f}")

    end = timeit.default_timer()

    return new_X