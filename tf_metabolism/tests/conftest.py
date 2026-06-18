from os.path import join, abspath, dirname

import pytest

data_model_dir = join(dirname(abspath(__file__)), 'data')
data_preprocessing_dir = join(dirname(abspath(__file__)), 'data_preprocessing')

@pytest.fixture(scope="function")
def model_file():
    model_file = join(data_model_dir, 'Recon2M.2_Entrez_Gene.xml')
    return model_file

@pytest.fixture(scope="function")
def expression_file():
    expression_file = join(data_preprocessing_dir, 'FPKM(adrenal_4a.V119).csv')
    return expression_file


@pytest.fixture(scope="function")
def expression_file_rank():
    expression_file_rank = join(data_preprocessing_dir, 'FPKM(adrenal_4a.V119)_rank_precalculated.csv')
    return expression_file_rank


@pytest.fixture(scope="function")
def expression_file_original():
    expression_file_original = join(data_preprocessing_dir, 'FPKM(adrenal_4a.V119)_original_precalculated.csv')
    return expression_file_original


@pytest.fixture(scope="function")
def reaction_weight_file():
    reaction_weight_file = join(data_preprocessing_dir, 'FPKM(adrenal_4a.V119)_rank_reaction_weight_precalculated.csv')
    return reaction_weight_file