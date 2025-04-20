import numpy as np

def suggest_model_parameters(rng, model_spec, num_suggestions=3):
    all_suggestions = []
    
    models = model_spec['values']
    
    for model in models:
        model_name = model['_name']
        
        for _ in range(num_suggestions):
            suggested_params = {}
            
            model_name_index = models.index(model)
            suggested_params[('model',)] = model_name_index  
            
            for param, param_spec in model.items():
                if param != '_name':
                    param_values = param_spec['_value']
                    param_index = rng.integers(0, len(param_values))
                    suggested_params[('model', model_name_index, param)] = param_index 
            
            all_suggestions.append(suggested_params)
    
    return all_suggestions

rng = np.random.default_rng() 

model_spec = {
    'name': 'model',
    'type': 'choice',
    # 'values': [
    #     {'_name': 'IF', 'z1_dim': {'_type': 'choice', '_value': [1, 2, 3]}, 'model_dim': {'_type': 'choice', '_value': [100, 200, 400]}, 'pretrain_lr': {'_type': 'choice', '_value': [0.008, 0.0008]}, 'train_lr': {'_type': 'choice', '_value': [0.05, 0.005, 0.0005]}, 'batch_size': {'_type': 'choice', '_value': [50, 100]}, 'epochs': {'_type': 'choice', '_value': [10, 20]}},
    #     {'_name': 'Omni', 'model_dim': {'_type': 'choice', '_value': [200, 400, 600]}, 'lr': {'_type': 'choice', '_value': [0.001, 0.0001, 1e-05]}, 'batch_size': {'_type': 'choice', '_value': [100, 200, 400]}, 'epochs': {'_type': 'choice', '_value': [10, 20, 30]}},
    #     {'_name': 'sdfvae', 's_dim': {'_type': 'choice', '_value': [4, 8, 12]}, 'd_dim': {'_type': 'choice', '_value': [5, 10, 15]}, 'model_dim': {'_type': 'choice', '_value': [50, 100, 150]}, 'lr': {'_type': 'choice', '_value': [0.001, 0.0001, 1e-05]}, 'batch_size': {'_type': 'choice', '_value': [64, 128]}, 'epochs': {'_type': 'choice', '_value': [10, 30]}},
    #     {'_name': 'Transformer', 'e_layers': {'_type': 'choice', '_value': [2, 4]}, 'd_layers': {'_type': 'choice', '_value': [1, 2]}, 'train_epochs': {'_type': 'choice', '_value': [3, 5]}, 'batch_size': {'_type': 'choice', '_value': [32, 64]}, 'learning_rate': {'_type': 'choice', '_value': [0.0001, 0.0002]}},
    #     {'_name': 'FEDformer', 'e_layers': {'_type': 'choice', '_value': [2, 4]}, 'd_layers': {'_type': 'choice', '_value': [1, 2]}, 'train_epochs': {'_type': 'choice', '_value': [3, 5]}, 'batch_size': {'_type': 'choice', '_value': [32, 64]}, 'learning_rate': {'_type': 'choice', '_value': [0.0001, 0.0002]}}
    # ],

     'values': [
        {'_name': 'IF', 'z1_dim': {'_type': 'choice', '_value': [1, 2, 3]}, 'model_dim': {'_type': 'choice', '_value': [100, 200, 400]}, 'pretrain_lr': {'_type': 'choice', '_value': [0.008, 0.0008]}, 'train_lr': {'_type': 'choice', '_value': [0.05, 0.005, 0.0005]}, 'batch_size': {'_type': 'choice', '_value': [50, 100]}, 'epochs': {'_type': 'choice', '_value': [10, 20]}},
        {'_name': 'Omni', 'model_dim': {'_type': 'choice', '_value': [200, 400, 600]}, 'lr': {'_type': 'choice', '_value': [0.001, 0.0001, 1e-05]}, 'batch_size': {'_type': 'choice', '_value': [100, 200, 400]}, 'epochs': {'_type': 'choice', '_value': [10, 20, 30]}},
        {'_name': 'sdfvae', 's_dim': {'_type': 'choice', '_value': [4, 8, 12]}, 'd_dim': {'_type': 'choice', '_value': [5, 10, 15]}, 'model_dim': {'_type': 'choice', '_value': [50, 100, 150]}, 'lr': {'_type': 'choice', '_value': [0.001, 0.0001, 1e-05]}, 'batch_size': {'_type': 'choice', '_value': [64, 128]}, 'epochs': {'_type': 'choice', '_value': [10, 30]}}
    ],

    'key': ('model',),
    'categorical': True,
    'size': 3
}

recommended_params = suggest_model_parameters(rng, model_spec, num_suggestions=3)

