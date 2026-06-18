import pyparsing


def convert_string_GPR_to_list_GPR(GPR_association):
    num = pyparsing.Word(pyparsing.alphanums)

    booleanop = pyparsing.oneOf('AND and OR or')
    expr = pyparsing.infixNotation(num,
                                   [(booleanop, 2, pyparsing.opAssoc.LEFT)])
    return expr.parseString(GPR_association)[0].asList()
