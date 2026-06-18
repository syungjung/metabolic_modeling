import pandas as pd
from cobra.io import read_sbml_model
from tf_metabolism.metabolic_model import model_editing
from tf_metabolism.omics_integration import INIT
from tf_metabolism.omics_integration import omics_score_calculation


class TestOmicsIntegration:
    """Test functions in tf_metabolism.omics_integration"""

    def test_omics_score_calculation(self, expression_file, expression_file_rank, expression_file_original):
        expression_df = pd.read_csv(expression_file, index_col=0)

        rank_precalculated_dict = {}
        with open(expression_file_rank, 'r') as fp:
            for line in fp:
                sptlist = line.strip().split(',')
                rank_precalculated_dict[sptlist[0].strip()] = float(sptlist[1].strip())

        original_precalculated_dict = {}
        with open(expression_file_original, 'r') as fp:
            for line in fp:
                sptlist = line.strip().split(',')
                original_precalculated_dict[sptlist[0].strip()] = float(sptlist[1].strip())

        rank_expression_dict = omics_score_calculation.calculate_rank_based_expression_score(expression_df, 0.25)
        for each_key in rank_expression_dict:
            assert str(rank_expression_dict[each_key])[0:5] == str(rank_precalculated_dict[each_key])[0:5]

        average_value = 58.4719832143
        original_score_expression_dict = omics_score_calculation.calculate_original_expression_score(expression_df,
                                                                                                     average_value)
        for each_key in original_score_expression_dict:
            assert str(original_score_expression_dict[each_key])[0:5] == str(original_precalculated_dict[each_key])[0:5]

    def test_GPR_score_calculation(self):
        expression_level_dic = {}
        expression_level_dic['a'] = 1.0
        expression_level_dic['b'] = 2.0
        expression_level_dic['c'] = 3.0
        expression_level_dic['d'] = 2.0
        expression_level_dic['e'] = 3.0
        expression_level_dic['f'] = 4.0
        expression_level_dic['g'] = 5.0
        expression_level_dic['h'] = 6.0
        expression_level_dic['i'] = 4.0

        case1_GPR_list = ['a', 'AND', 'b', 'OR', 'c', 'OR',
                          ['d', 'AND', 'e', 'OR', ['f', 'AND', ['g', 'OR', 'h', 'OR', 'i']]]]
        case2_GPR_list = ['a', 'OR', 'b', 'OR', 'c']
        case3_GPR_list = ['a', 'OR', 'b', 'AND', 'c']
        case4_GPR_list = [['a', 'OR', 'b'], 'AND', ['c', 'OR', 'd']]
        case5_GPR_list = [['a', 'AND', 'b'], 'AND', ['c', 'OR', 'd']]

        assert omics_score_calculation.GPR_score_calculation(case1_GPR_list, expression_level_dic) == 4.0
        assert omics_score_calculation.GPR_score_calculation(case2_GPR_list, expression_level_dic) == 3.0
        assert omics_score_calculation.GPR_score_calculation(case3_GPR_list, expression_level_dic) == 3.0
        assert omics_score_calculation.GPR_score_calculation(case4_GPR_list, expression_level_dic) == 2.0
        assert omics_score_calculation.GPR_score_calculation(case5_GPR_list, expression_level_dic) == 1.0