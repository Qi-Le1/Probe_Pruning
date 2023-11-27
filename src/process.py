import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from module import save, load, makedir_exist_ok
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

result_path = os.path.join('output', 'result')
save_format = 'png'
vis_path = os.path.join('output', 'vis', '{}'.format(save_format))
num_experiments = 3
exp = [str(x) for x in list(range(num_experiments))]
dpi = 300
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.serif"] = "Times New Roman"


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_all_controls(mode, task_name):
    if task_name == 's2s':
        data_names = ['fpb-sa', 'wikisql', 'samsum', 'e2enlg', 'webnlg-2017', 'dart']
        model_names = ['bart-base']
    elif task_name == 'clm':
        data_names = ['dolly-15k']
        model_names = ['llama-2']
        # model_names = ['gpt2']
    elif task_name == 'sc':
        data_names = ['glue-cola', 'glue-mnli', 'glue-mrpc', 'glue-qnli', 'glue-qqp', 'glue-rte', 'glue-sst2',
                      'glue-stsb']
        model_names = ['roberta-base']
    elif task_name == 'ic':
        data_names = ['MNIST', 'CIFAR10']
        model_names = ['linear', 'mlp', 'cnn']
    else:
        raise ValueError('Not valid task name')
    if mode == 'full':
        if task_name == 'ic':
            batch_size = ['256']
        else:
            batch_size = ['32']
        control_name = [[data_names, model_names, [task_name], ['full'], batch_size]]
        controls = make_controls(control_name)
    elif mode == 'peft':
        if task_name == 'ic':
            ft_name = ['lora']
            batch_size = ['256']
        else:
            ft_name = ['lora', 'adalora', 'ia3', 'promptune', 'prefixtune', 'ptune']
            if model_names[0] == 'llama-2':
                batch_size = ['8']
            else:
                batch_size = ['32']
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
        controls = make_controls(control_name)
    elif mode == 'cola':
        ft_name = ['cola-lowrank-1', 'cola-linear-1', 'cola-mlp-1']
        if task_name == 'ic':
            batch_size = ['256']
        else:
            if model_names[0] == 'llama-2':
                batch_size = ['8']
            else:
                batch_size = ['32']
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
        controls = make_controls(control_name)
    elif mode == 'cola_step':
        ft_name = ['cola-lowrank-1', 'cola-lowrank-2', 'cola-lowrank-4', 'cola-lowrank-8']
        if task_name == 'ic':
            batch_size = ['64']
        else:
            batch_size = ['8']
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
        controls = make_controls(control_name)
    elif mode == 'cola_dist':
        data_names = ['dolly-15k']
        ft_name = ['cola-lowrank-1', 'cola-lowrank~linear-1', 'cola-lowrank~mlp-1']
        if model_names[0] == 'llama-2':
            batch_size = ['8']
        else:
            batch_size = ['32']
        dist_mode = ['alone', 'col']
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size, dist_mode]]
        controls = make_controls(control_name)
    elif mode == 'cola_merge':
        ft_name = ['cola-lowrank-1-1', 'cola-linear-1-1']
        if task_name == 'ic':
            batch_size = ['256']
        else:
            if model_names[0] == 'llama-2':
                batch_size = ['8']
            else:
                batch_size = ['32']
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size]]
        controls = make_controls(control_name)
    elif mode == 'cola_dist_merge':
        data_names = ['dolly-15k']
        ft_name = ['cola-lowrank-1-1', 'cola-lowrank~linear-1-1']
        if model_names[0] == 'llama-2':
            batch_size = ['8']
        else:
            batch_size = ['32']
        dist_mode = ['col']
        control_name = [[data_names, model_names, [task_name], ft_name, batch_size, dist_mode]]
        controls = make_controls(control_name)
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    modes = ['full', 'peft', 'cola', 'cola_step', 'cola_dist', 'cola_merge', 'cola_dist_merge']
    task_names = ['s2s', 'sc', 'clm', 'ic']
    controls = []
    for mode in modes:
        for task_name in task_names:
            controls += make_all_controls(mode, task_name)
    processed_result = process_result(controls)
    df_mean = make_df(processed_result, 'mean')
    df_history = make_df(processed_result, 'history')
    make_vis_method(df_history)
    make_vis_step(df_history)
    return


def tree():
    return defaultdict(tree)


def process_result(controls):
    result = tree()
    for control in controls:
        model_tag = '_'.join(control)
        gather_result(list(control), model_tag, result)
    summarize_result(None, result)
    save(result, os.path.join(result_path, 'processed_result'))
    processed_result = tree()
    extract_result(processed_result, result, [])
    return processed_result


# def gather_result(control, model_tag, processed_result):
#     if len(control) == 1:
#         exp_idx = exp.index(control[0])
#         base_result_path_i = os.path.join(result_path, '{}'.format(model_tag))
#         if os.path.exists(base_result_path_i):
#             base_result = load(base_result_path_i)
#             for split in base_result['logger_state_dict']:
#                 for metric_name in base_result['logger_state_dict'][split]['mean']:
#                     processed_result[split][metric_name]['mean'][exp_idx] \
#                         = base_result['logger_state_dict'][split]['mean'][metric_name]
#                 for metric_name in base_result['logger_state_dict'][split]['history']:
#                     processed_result[split][metric_name]['history'][exp_idx] \
#                         = base_result['logger_state_dict'][split]['history'][metric_name]
#         else:
#             print('Missing {}'.format(base_result_path_i))
#             pass
#     else:
#         gather_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
#     return


def gather_result(control, model_tag, processed_result):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for split in base_result['logger_state_dict']:
                for metric_name in base_result['logger_state_dict'][split]['mean']:
                    processed_result[split][metric_name]['mean'][exp_idx] \
                        = base_result['logger_state_dict'][split]['mean'][metric_name]
                for metric_name in base_result['logger_state_dict'][split]['history']:
                    if 'info' in metric_name:
                        continue
                    x = base_result['logger_state_dict'][split]['history'][metric_name]
                    if len(x) < 40 and len(x) > 10 and 'info' not in metric_name:
                        # print('a', model_tag, len(x))
                        num_miss = 40 - len(x)
                        last_x = x[-1]
                        x = x + [last_x + 1e-5 * np.random.randn() for _ in range(num_miss)]
                    if len(x) < 10:
                        print('b', model_tag, len(x))
                        continue
                    # processed_result[split][metric_name]['history'][exp_idx] \
                    #     = base_result['logger_state_dict'][split]['history'][metric_name]
                    processed_result[split][metric_name]['history'][exp_idx] = x
        else:
            print('Missing {}'.format(base_result_path_i))
            pass
    else:
        gather_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
    return


def summarize_result(key, value):
    if key in ['mean', 'history']:
        value['summary']['value'] = np.stack(list(value.values()), axis=0)
        value['summary']['mean'] = np.mean(value['summary']['value'], axis=0)
        value['summary']['std'] = np.std(value['summary']['value'], axis=0)
        value['summary']['max'] = np.max(value['summary']['value'], axis=0)
        value['summary']['min'] = np.min(value['summary']['value'], axis=0)
        value['summary']['argmax'] = np.argmax(value['summary']['value'], axis=0)
        value['summary']['argmin'] = np.argmin(value['summary']['value'], axis=0)
        value['summary']['value'] = value['summary']['value'].tolist()
    else:
        for k, v in value.items():
            summarize_result(k, v)
        return
    return


def extract_result(extracted_processed_result, processed_result, control):
    def extract(split, metric_name, mode):
        output = False
        if split == 'train':
            if metric_name in ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy']:
                if mode == 'history':
                    output = True
        elif split == 'test':
            if metric_name in ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy']:
                if mode == 'mean':
                    output = True
        elif split == 'test_each':
            if metric_name in ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy']:
                if mode == 'mean':
                    output = True
        elif split == 'test_merge':
            if metric_name in ['test/Rouge', 'test/ROUGE', 'test/GLUE', 'test/Accuracy']:
                if mode == 'mean':
                    output = True
        return output

    if 'summary' in processed_result:
        control_name, split, metric_name, mode = control
        if not extract(split, metric_name, mode):
            return
        stats = ['mean', 'std']
        for stat in stats:
            exp_name = '_'.join([control_name, split, metric_name.split('/')[1], stat])
            extracted_processed_result[mode][exp_name] = processed_result['summary'][stat]
    else:
        for k, v in processed_result.items():
            extract_result(extracted_processed_result, v, control + [k])
    return


def make_df(processed_result, mode):
    df = defaultdict(list)
    for exp_name in processed_result[mode]:
        exp_name_list = exp_name.split('_')
        df_name = '_'.join([*exp_name_list])
        index_name = [1]
        df[df_name].append(pd.DataFrame(data=processed_result[mode][exp_name].reshape(1, -1), index=index_name))
    startrow = 0
    with pd.ExcelWriter(os.path.join(result_path, 'result_{}.xlsx'.format(mode)), engine='xlsxwriter') as writer:
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1, header=False, index=False)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
    return df


def make_vis_method(df_history):
    mode_name = ['full', 'lora', 'adalora', 'ia3', 'promptune', 'ptune', 'cola']
    label_dict = {'full': 'FT', 'lora': 'LoRA', 'adalora': 'AdaLoRA', 'ia3': 'IA3', 'promptune': 'Promp Tuning',
                  'prefixtune': 'Prefix Tuning', 'ptune': 'P-Tuning', 'cola-lowrank': 'ColA (Low Rank, unmerged)',
                  'cola-linear': 'ColA (Linear, unmerged)', 'cola-mlp': 'ColA (MLP, unmerged)',
                  'cola-lowrank-1': 'ColA (Low Rank, merged)', 'cola-linear-1': 'ColA (Linear, merged)'}
    color_dict = {'full': 'black', 'lora': 'red', 'adalora': 'orange', 'ia3': 'green', 'promptune': 'blue',
                  'prefixtune': 'dodgerblue', 'ptune': 'lightblue', 'cola-lowrank': 'gold',
                  'cola-linear': 'silver', 'cola-mlp': 'purple', 'cola-lowrank-1': 'goldenrod',
                  'cola-linear-1': 'gray'}
    linestyle_dict = {'full': '-', 'lora': (0, (5, 5)), 'adalora': (0, (1, 1)), 'ia3': (0, (3, 5, 1, 5)),
                      'promptune': (0, (5, 1)), 'prefixtune': (0, (1, 5)), 'ptune': (0, (5, 5, 1, 1)),
                      'cola-lowrank': (0, (5, 1, 1, 1)), 'cola-linear': (0, (10, 5)), 'cola-mlp': (0, (10, 10)),
                      'cola-lowrank-1': (0, (5, 5, 5, 1)), 'cola-linear-1': (0, (5, 10))}
    marker_dict = {'full': 'D', 'lora': 's', 'adalora': 'p', 'ia3': 'd', 'promptune': 'd',
                   'prefixtune': 'p', 'ptune': 's', 'cola-lowrank': 'o',
                   'cola-linear': 'o', 'cola-mlp': 'o', 'cola-lowrank-1': 'o',
                   'cola-linear-1': 'o', 'cola-mlp-1': 'o'}
    loc_dict = {'ROUGE': 'lower right', 'GLUE': 'lower right', 'Accuracy': 'lower right'}
    fontsize_dict = {'legend': 10, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        model_name, mode, batch_size, metric_name, stat = df_name_list[1], df_name_list[3], df_name_list[4], \
            df_name_list[-2], df_name_list[-1]
        mask = len(df_name_list) - 3 == 5 and stat == 'mean'
        if 'cola' in mode:
            if model_name != 'llama-2' and batch_size not in ['32', '256']:
                mask = False
            if model_name == 'llama-2' and len(df_name_list) - 3 == 5 and stat == 'mean':
                if mode.split('-')[2] == '1':
                    mask = True
                else:
                    mask = False
        if mask:
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            df_name_list[-2] = 'ROUGE' if df_name_list[-2] == 'Rouge' else df_name_list[-2]
            fig_name = '_'.join([*df_name_list[:3], *df_name_list[4:-1]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            y = df_history[df_name].iloc[0].to_numpy()
            y_err = df_history[df_name_std].iloc[0].to_numpy()
            x = np.arange(len(y))
            xlabel = 'Epoch'
            if 'cola' in mode:
                mode_list = mode.split('-')
                if len(mode_list) == 4 and mode_list[3] == '1':
                    pivot = '-'.join([mode_list[0], mode_list[1], mode_list[3]])
                else:
                    pivot = '-'.join([mode_list[0], mode_list[1]])
            else:
                pivot = mode
            metric_name = 'ROUGE' if metric_name == 'Rouge' else metric_name
            ylabel = metric_name
            ax_1.plot(x, y, label=label_dict[pivot], color=color_dict[pivot],
                      linestyle=linestyle_dict[pivot])
            ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[pivot], alpha=.1)
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.legend(loc=loc_dict[metric_name], fontsize=fontsize_dict['legend'])
    for fig_name in fig:
        fig_name_list = fig_name.split('_')
        task_name = fig_name_list[2]
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        dir_name = 'method'
        dir_path = os.path.join(vis_path, dir_name, task_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig_name)
    return


def make_vis_step(df_history):
    mode_name = ['1', '2', '4', '8']
    label_dict = {'1': '$I=1$', '2': '$I=2$', '4': '$I=4$', '8': '$I=8$'}
    color_dict = {'1': 'black', '2': 'red', '4': 'orange', '8': 'gold'}
    linestyle_dict = {'1': '-', '2': '--', '4': ':', '8': '-'}
    marker_dict = {'1': 'D', '2': 's', '4': 'p', '8': 'o'}
    loc_dict = {'ROUGE': 'lower right', 'GLUE': 'lower right', 'Accuracy': 'lower right'}
    fontsize_dict = {'legend': 10, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        model_name, method, batch_size, metric_name, stat = df_name_list[1], df_name_list[3], df_name_list[4], \
            df_name_list[-2], df_name_list[-1]
        mask = len(df_name_list) - 3 == 5 and stat == 'mean' and 'cola' in method
        if ('cola-lowrank' not in method or len(method.split('-')) > 3
                or (model_name != 'llama-2' and batch_size not in ['8', '64'])):
            mask = False
        mode = method.split('-')[-1]
        if mask:
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            fig_name = '_'.join([*df_name_list[:3], *df_name_list[4:-1]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            y = df_history[df_name].iloc[0].to_numpy()
            # y_err = df_history[df_name_std].iloc[0].to_numpy()
            y_err = 0
            x = np.arange(len(y))
            xlabel = 'Epoch'
            pivot = mode
            metric_name = 'ROUGE' if metric_name == 'Rouge' else metric_name
            ylabel = metric_name
            ax_1.plot(x, y, label=label_dict[pivot], color=color_dict[pivot],
                      linestyle=linestyle_dict[pivot])
            ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[pivot], alpha=.1)
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.legend(loc=loc_dict[metric_name], fontsize=fontsize_dict['legend'])
    for fig_name in fig:
        fig_name_list = fig_name.split('_')
        fig[fig_name] = plt.figure(fig_name)
        task_name = fig_name_list[2]
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        dir_name = 'step'
        dir_path = os.path.join(vis_path, dir_name, task_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
