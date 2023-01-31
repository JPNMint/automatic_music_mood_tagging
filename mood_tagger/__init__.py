import os
import typing

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
ARCH_PATH = os.path.join(ROOT_PATH, 'architectures')


def get_architecture(architecture_name: str) -> typing.Tuple[typing.Type, str]:
    """ select architecture """
    architecture_file = os.path.join(ARCH_PATH, architecture_name + '.py')
    namespace = dict()
    namespace['architecture'] = None
    exec(f'from mood_tagger.architectures.{architecture_name} import Net as architecture', namespace)
    return namespace['architecture'], architecture_file