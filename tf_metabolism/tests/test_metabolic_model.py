from cobra.io import read_sbml_model
from tf_metabolism.metabolic_model import GPR_manipulation


class TestMetabolicModel:
    """Test functions in tf_metabolism.metabolic_simulation.Simulator"""

    def test_GPR_manipuation(self):
        case1 = GPR_manipulation.convert_string_GPR_to_list_GPR("(a AND b) OR (c OR d OR e)")
        case2 = GPR_manipulation.convert_string_GPR_to_list_GPR("(a AND b) OR (c OR d AND e)")
        case3 = GPR_manipulation.convert_string_GPR_to_list_GPR("(a AND b) OR c OR d AND e")

        assert case1 == [['a', 'AND', 'b'], 'OR', ['c', 'OR', 'd', 'OR', 'e']]
        assert case2 == [['a', 'AND', 'b'], 'OR', ['c', 'OR', 'd', 'AND', 'e']]
        assert case3 == [['a', 'AND', 'b'], 'OR', 'c', 'OR', 'd', 'AND', 'e']
