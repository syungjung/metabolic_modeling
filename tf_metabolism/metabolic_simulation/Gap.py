import copy

from gurobipy import *
from tf_metabolism.metabolic_simulation import Simulator


class GapFilling(Simulator.Simulator):
    def __init__(self):
        """
        Constructor for Gap-filling
        """
        self.threshold = 0.0001

    def run_GapFill(self, target_reaction, flux_constraints={}, universal_reactions=[], inf_flag=False,
                    limit_number=99999):
        added_reactions = []

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

        m = Model('Gap filling')
        m.setParam('OutputFlag', 0)
        # m.setParam('DualReductions', 0)
        m.reset()

        # create variables
        epsilon = 0.0001

        v = {}
        fplus = {}
        fminus = {}
        b_bool = {}

        for each_reaction in model_reactions:
            v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction],
                                        ub=upper_boundary_constraints[each_reaction], name=each_reaction)
            fplus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)
            fminus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)

        for each_reaction in model_reactions:
            b_bool[each_reaction] = m.addVar(vtype=GRB.BINARY, name=each_reaction)

        m.update()

        for each_reaction in model_reactions:
            m.addConstr(v[each_reaction] == (fplus[each_reaction] - fminus[each_reaction]))

        for each_reaction in model_reactions:
            m.addConstr((fplus[each_reaction] - fminus[each_reaction]) >= lower_boundary_constraints[each_reaction])
            m.addConstr((fplus[each_reaction] - fminus[each_reaction]) <= upper_boundary_constraints[each_reaction])

        for each_reaction in universal_reactions:
            m.addConstr((fplus[each_reaction] - fminus[each_reaction]) <= 1000.0 * b_bool[each_reaction])
            m.addConstr(
                (fplus[each_reaction] - fminus[each_reaction]) + 1000.0 * (1 - b_bool[each_reaction]) >= epsilon)

        m.addConstr(quicksum((b_bool[each_reaction]) for each_reaction in universal_reactions) <= limit_number)

        m.addConstr((fplus[target_reaction] - fminus[target_reaction]) >= self.threshold)

        m.update()

        # Add constraints
        for each_metabolite in model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(quicksum(
                (fplus[each_reaction] - fminus[each_reaction]) * coffvalue[metabolite, each_reaction] for
                metabolite, each_reaction in
                pairs.select(each_metabolite, '*')) == 0)

        m.update()

        m.setObjective(quicksum((b_bool[each_reaction]) for each_reaction in universal_reactions), GRB.MINIMIZE)
        m.optimize()

        if m.status == 2:
            for reaction in universal_reactions:
                if b_bool[reaction].x > 0.0:
                    added_reactions.append(reaction)
            return m.status, m.ObjVal, added_reactions
        else:
            return m.status, False, False

    def load_universal_model(self, universal_model):
        self.universal_model = universal_model

    def fill_gap(self, target_reaction):
        universal_model = self.universal_model
        cobra_model = self.cobra_model

        cobra_reactions = [each_reaction.id for each_reaction in self.cobra_model.reactions]
        added_reactions = []
        universal_reactions = []

        for each_reaction in universal_model.reactions:
            if each_reaction.id not in cobra_reactions:
                added_reactions.append(each_reaction)
                universal_reactions.append(each_reaction.id)

        self.cobra_model.add_reactions(added_reactions)
        self.load_cobra_model(self.cobra_model)

        model_status, objective_value, added_reactions = self.run_GapFill(target_reaction=target_reaction,
                                                                          universal_reactions=universal_reactions)

        return model_status, objective_value, added_reactions

    def run_gap_filling(self, universal_model, metabolic_gap_model, target_reaction):
        self.load_universal_model(copy.deepcopy(universal_model))
        self.load_cobra_model(copy.deepcopy(metabolic_gap_model))

        model_status, objective_value, added_reactions = self.fill_gap(target_reaction)
        if model_status == 2:
            reaction_object_list = []
            for each_reaction in added_reactions:
                reaction_object_list.append(self.universal_model.reactions.get_by_id(each_reaction))
            return model_status, objective_value, added_reactions, reaction_object_list
        else:
            return model_status, False, False, False

    def set_threshold(self, threshold):
        self.threshold = threshold
