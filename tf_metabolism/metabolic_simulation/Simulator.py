import cobra.io as io
from gurobipy import *


class Simulator(object):
    def __init__(self):
        """
        Constructor for Simulator
        """
        self.cobra_model = None
        self.model_metabolites = None
        self.model_reactions = None
        self.Smatrix = None
        self.lower_boundary_constraints = None
        self.upper_boundary_constraints = None
        self.objective = None

    def run_MOMA(self, wild_flux={}, flux_constraints={}, inf_flag=False):
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

        m = Model('MOMA')
        m.setParam('OutputFlag', 0)
        m.reset()

        # create variables
        target_reactions = wild_flux.keys()
        v = {}

        for each_reaction in model_reactions:
            if each_reaction in flux_constraints:
                v[each_reaction] = m.addVar(lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1], name=f'v_{each_reaction}')
            else:
                v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction],
                                            ub=upper_boundary_constraints[each_reaction],
                                            name=f'v_{each_reaction}')

        m.update()

        # Stoichiometric constraints: S·v = 0
        for each_metabolite in model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(quicksum(
                v[reaction] * coffvalue[metabolite, reaction] for metabolite, reaction in
                pairs.select(each_metabolite, '*')) == 0)

        m.update()

        # L2-MOMA objective: min Σ (v_KO[r] - v_WT[r])²
        m.setObjective(
            quicksum((v[r] - wild_flux[r]) * (v[r] - wild_flux[r]) for r in target_reactions),
            GRB.MINIMIZE)

        m.optimize()

        if m.status == 2:
            flux_distribution = {}
            for reaction in model_reactions:
                flux_distribution[reaction] = float(v[reaction].x)
                if abs(float(v[reaction].x)) <= 1e-6:
                    flux_distribution[reaction] = 0.0

            return m.status, m.ObjVal, flux_distribution
        else:
            return m.status, False, False

    def run_FBA(self, new_objective='', flux_constraints={}, inf_flag=False, internal_flux_minimization=False,
                mode='max'):
        model_metabolites = self.model_metabolites
        model_reactions = self.model_reactions

        if new_objective == '':
            objective = self.objective
        else:
            objective = new_objective

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

        m = Model('FBA')
        m.setParam('OutputFlag', 0)
        m.reset()

        # create variables
        v = {}
        fplus = {}
        fminus = {}

        m.update()

        for each_reaction in model_reactions:
            if each_reaction in flux_constraints:
                v[each_reaction] = m.addVar(lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1], name=f'v_{each_reaction}')
            else:
                v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction],
                                            ub=upper_boundary_constraints[each_reaction],
                                            name=f'v_{each_reaction}')
            fplus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=f'fp_{each_reaction}')
            fminus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=f'fm_{each_reaction}')
        m.update()

        # Add constraints
        for each_metabolite in model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(quicksum(v[reaction] * coffvalue[metabolite, reaction] for metabolite, reaction in
                                 pairs.select(each_metabolite, '*')) == 0)

        m.update()
        if mode == 'max':
            m.setObjective(v[objective], GRB.MAXIMIZE)
        elif mode == 'min':
            m.setObjective(v[objective], GRB.MINIMIZE)

        m.optimize()
        if m.status == 2:
            objective_value = m.ObjVal
            if internal_flux_minimization:
                m.addConstr(fplus[objective] - fminus[objective] == objective_value)

                m.addConstr(quicksum(
                    (fplus[reaction] - fminus[reaction]) * coffvalue[metabolite, reaction] for metabolite, reaction in
                    pairs.select(each_metabolite, '*')) == 0)

                for each_reaction in model_reactions:
                    m.addConstr(fplus[each_reaction] - fminus[each_reaction] == v[each_reaction])

                m.update()
                m.setObjective(
                    quicksum((fplus[each_reaction] + fminus[each_reaction]) for each_reaction in model_reactions),
                    GRB.MINIMIZE)
                m.optimize()
                if m.status == 2:
                    objective_value = m.ObjVal
                    flux_distribution = {}
                    for reaction in model_reactions:
                        flux_distribution[reaction] = float(v[reaction].x)
                        if abs(float(v[reaction].x)) <= 1e-6:
                            flux_distribution[reaction] = 0.0
                    return m.status, objective_value, flux_distribution
            else:
                flux_distribution = {}
                for reaction in model_reactions:
                    flux_distribution[reaction] = float(v[reaction].x)
                    if abs(float(v[reaction].x)) <= 1e-6:
                        flux_distribution[reaction] = 0.0
                return m.status, objective_value, flux_distribution
        return m.status, False, False

    def read_model(self, filename):
        model = io.read_sbml_model(filename)
        return self.load_cobra_model(model)

    def load_cobra_model(self, cobra_model):
        self.cobra_model = cobra_model
        model = cobra_model
        model_metabolites = []
        model_reactions = []
        model_genes = []
        lower_boundary_constraints = {}
        upper_boundary_constraints = {}
        objective_reaction = ''
        for each_metabolite in model.metabolites:
            model_metabolites.append(each_metabolite.id)

        model_genes = [each_gene.id for each_gene in model.genes]

        Smatrix = {}

        for each_reaction in model.reactions:
            if each_reaction.objective_coefficient == 1.0:
                objective_reaction = each_reaction.id

            reactant_list = each_reaction.reactants
            reactant_coff_list = each_reaction.get_coefficients(reactant_list)
            product_list = each_reaction.products
            product_coff_list = each_reaction.get_coefficients(product_list)

            for reactant, coeff in zip(reactant_list, reactant_coff_list):
                Smatrix[(reactant.id, each_reaction.id)] = coeff
                
            for product, coeff in zip(product_list, product_coff_list):
                Smatrix[(product.id, each_reaction.id)] = coeff

            model_reactions.append(each_reaction.id)
            lb = each_reaction.lower_bound
            ub = each_reaction.upper_bound
            if lb < -1000.0:
                lb = float('-inf')
            if ub > 1000.0:
                ub = float('inf')
            lower_boundary_constraints[each_reaction.id] = lb
            upper_boundary_constraints[each_reaction.id] = ub

        self.model_metabolites = model_metabolites
        self.model_reactions = model_reactions
        self.model_genes = model_genes
        self.Smatrix = Smatrix
        self.lower_boundary_constraints = lower_boundary_constraints
        self.upper_boundary_constraints = upper_boundary_constraints
        self.objective = objective_reaction

        return (model_metabolites, model_reactions, Smatrix, lower_boundary_constraints, upper_boundary_constraints,
                objective_reaction)


if __name__ == '__main__':
    obj = Simulator()
    obj.read_model(
        '/data2/jupyter_work/jyryu3161/git_work_dir/reconmanagement/management/tests/data/Recon2M.2_Entrez_Gene.xml')
    a, b, c = obj.run_FBA()
    print(c['biomass_reaction'])
    print(len(obj.model_reactions))
    print(len(obj.model_metabolites))
    print(len(obj.model_genes))
