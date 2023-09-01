import os
import typing

import numpy as np
import torch
import importlib.util

from mood_tagger.data import GEMS_9, plot_data, generate_dataframe

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
ARCH_PATH = os.path.join(ROOT_PATH, 'architectures')


def get_architecture(architecture_name: str, from_path: str = None) -> typing.Tuple[typing.Type, str]:
    """ select architecture """
    arch_file_name = architecture_name + '.py'
    if from_path is None:
        architecture_file = os.path.join(ARCH_PATH, arch_file_name)
        namespace = dict()
        namespace['architecture'] = None
        exec(f'from mood_tagger.architectures.{architecture_name} import Net as architecture', namespace)
        return namespace['architecture'], architecture_file
    else:
        architecture_file = os.path.join(from_path, arch_file_name)
        arch_spec = importlib.util.spec_from_file_location(architecture_name, architecture_file)
        arch_module = importlib.util.module_from_spec(arch_spec)
        arch_spec.loader.exec_module(arch_module)
        return arch_module.Net, architecture_file


def test_model(model, num_classes, test_loader, device, plot=False, model_name=None):
    num_examples = len(test_loader)
    mean_abs_errors = np.zeros((num_classes,))
    mean_errors = np.zeros((num_classes,))
    maximums = np.zeros((num_classes,))
    minimums = np.ones((num_classes,)) * 100
    model.to(device)
    predictions = []
    prediction_names = []
    prediction_genres = []
    annotations = []
    for test_feat, test_targ, test_annot, test_name, test_genre in test_loader:
        # print(test_name[0])
        with torch.no_grad():
            test_feat = test_feat.to(device)
            model_outs = model(test_feat[:, :, 0]).cpu().numpy()[0]
            error = test_targ[0].numpy() - model_outs

        annotations.append(test_targ.cpu().numpy()[0])
        predictions.append(model_outs)
        prediction_names.append(test_name[0])
        prediction_genres.append(test_genre[0])

        mean_abs_errors += np.abs(error)
        mean_errors += error
        maximums = np.maximum(maximums, model_outs)
        minimums = np.minimum(minimums, model_outs)
    mean_abs_errors /= num_examples
    mean_errors /= num_examples

    def ff(array):
        return np.array2string(array, precision=2, separator=' \t ', suppress_small=True)

    print(f"for {num_examples} test samples:\n"
          f"          {' '.join(GEMS_9)} \n"
          f"mean err: {ff(mean_errors)} \n"
          f"m abs er: {ff(mean_abs_errors)} \n"
          f"maximums: {ff(maximums)} \n"
          f"minimums: {ff(minimums)} ")

    print(f"\noverall: \nME: {np.mean(mean_errors):.2f} \nMAE: {np.mean(mean_abs_errors):.2f}")

    if plot:
        df_pred = generate_dataframe(predictions, prediction_genres)
        plot_data(df_pred, f'Test Predictions {model_name}')

        df_an = generate_dataframe(annotations, prediction_genres)
        plot_data(df_an, f'Test Annotations')
        print('plotted')
        pass
