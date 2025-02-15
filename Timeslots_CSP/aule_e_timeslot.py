import constraint as csp

time_solts = [1, 2, 3, 4, 5, 6, 7, 8, 9]

def crea_problema(lezioni, aule, time_slots):
    problema = csp.Problem()

    possible_values = [(aula, time_slots) for aula in aule]
    for lezione, i in lezioni:
        problema.addVariable(f"lezione_{i}", possible_values)