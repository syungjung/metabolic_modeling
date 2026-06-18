import copy

from gurobipy import *
from tf_metabolism import utils
from tf_metabolism.metabolic_simulation import Simulator


class INIT(Simulator.Simulator):
    def __init__(self):
        '''
        Constructor
        '''

    def INIT(self, weight_vectors={}, present_metabolites=[], flux_constraints={}, essential_reactions=[],
             inf_flag=False):
        model_metabolites = self.model_metabolites
        model_reactions = self.model_reactions

        Smatrix = self.Smatrix

        lower_boundary_constraints = self.lower_boundary_constraints
        upper_boundary_constraints = self.upper_boundary_constraints

        if not inf_flag:
            for key in lower_boundary_constraints:
                if lower_boundary_constraints[key] == float("-inf"):
                    lower_boundary_constraints[key] = -1000.0

            for key in upper_boundary_constraints:
                if upper_boundary_constraints[key] == float("inf"):
                    upper_boundary_constraints[key] = 1000.0

        pairs, coffvalue = multidict(Smatrix)
        pairs = tuplelist(pairs)

        m = Model('INIT')
        m.setParam('OutputFlag', 0)
        m.setParam('DualReductions', 0)
        m.reset()

        # create variables
        v = {}
        b = {}

        fplus = {}
        fminus = {}

        y_bool = {}
        x_bool = {}

        epsilon = 0.0001

        # Add variables
        for each_reaction in model_reactions:
            v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction],
                                        ub=upper_boundary_constraints[each_reaction], name=each_reaction)
            fplus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)
            fminus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)

        for each_reaction in weight_vectors:
            y_bool[each_reaction] = m.addVar(vtype=GRB.BINARY, name=each_reaction)

        m.update()

        for each_reaction in model_reactions:
            m.addConstr(v[each_reaction] == (fplus[each_reaction] - fminus[each_reaction]))

        for each_reaction in model_reactions:
            m.addConstr((fplus[each_reaction] - fminus[each_reaction]) >= lower_boundary_constraints[each_reaction])
            m.addConstr((fplus[each_reaction] - fminus[each_reaction]) <= upper_boundary_constraints[each_reaction])

        for each_metabolite in present_metabolites:
            b[each_metabolite] = m.addVar(lb=0.0, ub=1000.0, name=each_metabolite)

        for each_metabolite in present_metabolites:
            x_bool[each_metabolite] = m.addVar(vtype=GRB.BINARY, name=each_metabolite)

        m.update()

        for each_reaction in essential_reactions:
            m.addConstr(y_bool[each_reaction] == 1)
            m.addConstr((fplus[each_reaction] - fminus[each_reaction]) >= epsilon)

        for each_reaction in weight_vectors:
            m.addConstr((fplus[each_reaction] - fminus[each_reaction]) <= 1000.0 * y_bool[each_reaction])
            m.addConstr(
                (fplus[each_reaction] - fminus[each_reaction]) + 1000.0 * (1 - y_bool[each_reaction]) >= epsilon)

        for each_metabolite in present_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(b[each_metabolite] <= 1000.0 * x_bool[each_metabolite])
            m.addConstr(b[each_metabolite] + 1000.0 * (1 - x_bool[each_metabolite]) >= epsilon)

        # Steady state
        for each_metabolite in model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue

            if each_metabolite in present_metabolites:
                m.addConstr(quicksum(
                    (fplus[each_reaction] - fminus[each_reaction]) * coffvalue[metabolite, each_reaction] for
                    metabolite, each_reaction in pairs.select(each_metabolite, '*')) == b[each_metabolite])
            else:
                m.addConstr(quicksum(
                    (fplus[each_reaction] - fminus[each_reaction]) * coffvalue[metabolite, each_reaction] for
                    metabolite, each_reaction in pairs.select(each_metabolite, '*')) == 0.0)
        m.update()

        m.setObjective(quicksum(
            (weight_vectors[each_reaction] * y_bool[each_reaction]) for each_reaction in weight_vectors) + quicksum(
            (x_bool[each_metabolite]) for each_metabolite in present_metabolites), GRB.MAXIMIZE)

        m.optimize()

        if m.status == 2:
            flux_distribution = {}
            reaction_boolean_information = {}
            for each_reaction in model_reactions:
                flux_distribution[each_reaction] = float(fplus[each_reaction].x) - float(fminus[each_reaction].x)
                if each_reaction in y_bool:
                    reaction_boolean_information[each_reaction] = float(y_bool[each_reaction].x)

            metabolite_cnt = 0
            for each_metabolite in present_metabolites:
                if float(x_bool[each_metabolite].x) > 0:
                    metabolite_cnt += 1
            return m.status, m.ObjVal, flux_distribution, reaction_boolean_information
        else:
            return m.status, False, False, False

    def run_INIT(self, weight_vectors={}, present_metabolites=[], flux_constraints={}, essential_reactions=[],
                 inf_flag=False):
        cobra_model = self.cobra_model
        model_status, objective_value, flux_distribution, reaction_boolean_information = self.INIT(
            weight_vectors=weight_vectors, present_metabolites=present_metabolites, flux_constraints=flux_constraints,
            essential_reactions=essential_reactions)
        if model_status == 2:
            active_metabolic_reactions = []
            for each_reaction in flux_distribution:
                if each_reaction in flux_distribution:
                    if abs(flux_distribution[each_reaction]) > 0:
                        active_metabolic_reactions.append(each_reaction)
                if each_reaction in essential_reactions:
                    active_metabolic_reactions.append(each_reaction)

            for each_reaction in cobra_model.reactions:
                if each_reaction.boundary:
                    active_metabolic_reactions.append(each_reaction.id)

            active_metabolic_reactions = list(set(active_metabolic_reactions))

            model_reactions = [reaction.id for reaction in cobra_model.reactions]
            removed_reactions = set(model_reactions).difference(set(active_metabolic_reactions))

            context_model = copy.deepcopy(cobra_model)
            context_model.remove_reactions(removed_reactions)
            context_model = utils.update_cobra_model(context_model)
            return model_status, objective_value, flux_distribution, context_model
        else:
            return model_status, False, False, False
