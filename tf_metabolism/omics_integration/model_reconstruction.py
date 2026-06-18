from tf_metabolism import utils
from tf_metabolism.omics_integration import INIT


def reconstruct_GEM(generic_cobra_model, reaction_weight, present_metabolites, essential_reactions, biomass_reaction):
    init_obj = INIT.INIT()
    init_obj.load_cobra_model(generic_cobra_model)
    model_status, objective_value, flux_distribution, context_model = init_obj.run_INIT(weight_vectors=reaction_weight,
                                                                                        present_metabolites=present_metabolites,
                                                                                        essential_reactions=[
                                                                                            biomass_reaction])
    if model_status == 2:
        context_model_reactions = [each_reaction.id for each_reaction in context_model.reactions]
        essential_reaction_list = []
        for each_reaction in generic_cobra_model.reactions:
            if each_reaction.id in essential_reactions and each_reaction.id not in context_model_reactions:
                essential_reaction_list.append(each_reaction)
        if len(essential_reaction_list) > 0:
            context_model.add_reactions(essential_reaction_list)
        context_model = utils.update_cobra_model(context_model)
        return model_status, objective_value, flux_distribution, context_model
    else:
        return model_status, False, False, False
