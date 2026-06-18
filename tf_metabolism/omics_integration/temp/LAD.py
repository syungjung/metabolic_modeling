from tf_metabolism.metabolic_simulation import Simulator
from gurobipy import *

class LAD(Simulator.Simulator):
    def __init__(self):
        '''
        Constructor
        '''

    def run_LP_fitting(self, opt_flux={}, flux_constraints={}, inf_flag=False):
        model_metabolites = self.model_metabolites
        model_reactions = self.model_reactions

        Smatrix = self.Smatrix

        lower_boundary_constraints = self.lower_boundary_constraints
        upper_boundary_constraints = self.upper_boundary_constraints

        if inf_flag == False:
            for key in lower_boundary_constraints.keys():
                if lower_boundary_constraints[key] == float("-inf"):
                    lower_boundary_constraints[key] = -1000.0

            for key in upper_boundary_constraints.keys():
                if upper_boundary_constraints[key] == float("inf"):
                    upper_boundary_constraints[key] = 1000.0

        pairs, coffvalue = multidict(Smatrix)
        pairs = tuplelist(pairs)

        m = Model('LAD')
        m.setParam('OutputFlag', 0)
        m.reset()

        # create variables
        target_reactions = opt_flux.keys()
        v = {}
        fplus = {}
        fminus = {}

        for each_reaction in model_reactions:
            if each_reaction in flux_constraints.keys():
                v[each_reaction] = m.addVar(lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1], name=each_reaction)
            else:
                v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction], ub=upper_boundary_constraints[each_reaction],
                                            name=each_reaction)
            fplus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)
            fminus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)

        m.update()

        for each_reaction in opt_flux:
            m.addConstr(v[each_reaction] == (fplus[each_reaction] - fminus[each_reaction]))
            m.addConstr(fplus[each_reaction], GRB.GREATER_EQUAL, v[each_reaction] - opt_flux[each_reaction],
                        name=each_reaction)
            m.addConstr(fminus[each_reaction], GRB.GREATER_EQUAL, opt_flux[each_reaction] - v[each_reaction],
                        name=each_reaction)

        m.update()

        # Add constraints
        for each_metabolite in model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(quicksum(
                (fplus[reaction] - fminus[reaction]) * coffvalue[metabolite, reaction] for metabolite, reaction in
                pairs.select(each_metabolite, '*')) == 0)

        m.update()

        m.setObjective(quicksum(
            ((fplus[each_reaction] + fminus[each_reaction]) - opt_flux[each_reaction]) for each_reaction in
            target_reactions), GRB.MINIMIZE)

        m.optimize()

        if m.status == 2:
            ReactionFlux = {}
            for reaction in model_reactions:
                ReactionFlux[reaction] = float(v[reaction].x)

            return m.status, m.ObjVal, ReactionFlux
        else:
            return m.status, False, False


def read_expression_data(filename):
    expression_info = {}
    fp = open(filename, 'r')
    fp.readline()
    for line in fp:
        sptlist = line.split('\t')
        gene_id = sptlist[0].strip()
        value = sptlist[1].strip()
        expression_info[gene_id] = float(value)
    fp.close()
    return expression_info


