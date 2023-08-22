import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import (
    fetch_dataset, 
    split_dataset, 
    make_data_loader, 
    separate_dataset, 
    make_batchnorm_dataset, 
    make_batchnorm_stats
)
from metrics import Metric
from models.api import make_batchnorm
# from utils import save, to_device, process_control, process_dataset, resume, collate
from utils.api import (
    save, 
    to_device, 
    process_command, 
    process_dataset,  
    resume, 
    collate
)

from models.api import (
    # CNN,
    create_model,
    make_batchnorm
)

from logger import make_logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_command()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    last_epoch = result['epoch']
    data_split = result['data_split']
    relu_threshold_list = [0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]
    for relu_threshold in relu_threshold_list:
        cfg['relu_threshold'] = relu_threshold
        model = create_model()
        model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
        batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
        metric = Metric({'test': ['Loss', 'Accuracy']})
        
        model.load_state_dict(result['server'].server_model_state_dict)
        data_loader = make_data_loader(dataset, 'server')
        test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'server')
        test(data_loader['test'], test_model, metric, train_logger, last_epoch, relu_threshold)

    
    result = {'cfg': cfg, 
              'logger': {'train': train_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return 1


def test(data_loader, model, metric, logger, epoch, relu_threshold):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            key_name = f'ReLU_{relu_threshold}'
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            accuracy = evaluation['Accuracy']
            evaluation = {key_name: accuracy}
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print('zheli---', flush=True)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return



if __name__ == "__main__":
    main()
