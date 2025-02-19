import constraint as csp

#lezioni = [{"nome":"nome_lazione", "durata":numero_timeslot_occupati, "studenti_previsti":numero_studenti_previsti}]
#aule = [{"nome":"nome_aula", "capienza":capienza_aula}]
#possible_values = [(aula, time_slot)]

#con questa funzione generiamo il dominio di ogni variabile: queste prendono tutti i valori possibili in base alla durata delle lezione, e in base alla lista dei timsslots
def crea_dominio(lezioni, aule, timeslot):
    domini = {}
    for item in lezioni:
        possible_values = []
        durata = item["durata"]
        for aula in aule:
            for i in range(len(timeslot) - durata + 1):
                possible_values.append((aula["nome_aula"], tuple(timeslot[i:i+durata])))
        domini[item["nome_lezione"]] = possible_values
    return domini

def vincolo_posti(assegnazione, nome_lezione, lezioni, aule):
    nome_aula,_ = assegnazione
    lezione = next((lez for lez in lezioni if lez["nome_lezione"] == nome_lezione), None)
    aula = next((a for a in aule if a["nome_aula"] == nome_aula), None)

    return aula["capienza"] >= lezione["studenti_previsti"]

#questo vincolo serve a non far sovrapporre lezioni negli stessi timeslot / aule
def vincolo_limita_domini(*assegnazioni):
    occupazione = {} #questi dizionario tiene traccia delle aule / timeslot occupati

    for assegnazione in assegnazioni:
        if assegnazione is None:
            continue  

        #assegnazione è una coppia (aula, timeslot)
        _, slots = assegnazione

        for slot in slots:
            if slot in occupazione:
                return False  #se lo slot si trova in occupazione allora viene ritornato False
            occupazione[slot] = True  
            #questo settaggio aiuta a capire quali slot sono occupati

    return True 

def resolver(lezioni, aule, timeslot):
    problema = csp.Problem()
    domini = crea_dominio(lezioni=lezioni, aule=aule, timeslot=timeslot) #timeslots = [1,2,...,9]

    for lez in lezioni:
        problema.addVariable(lez["nome_lezione"], domini[lez["nome_lezione"]])
    
    for lezione in lezioni:
        problema.addConstraint(lambda assegnazione, l=lezione["nome_lezione"]: vincolo_posti(assegnazione=assegnazione, nome_lezione=l, lezioni=lezioni, aule=aule), [lezione["nome_lezione"]])
        
    problema.addConstraint(vincolo_limita_domini, tuple(lezione["nome_lezione"] for lezione in lezioni))

    #print("Variabili: ", problema._variables)
    #print("Vincoli: ", problema._constraints)

    soluzione = problema.getSolution()
    return soluzione

timeslots = [1, 2, 3, 4, 5, 6, 7, 8, 9]

lezioni = [
        {"nome_lezione":"Analisi 1", "durata":2, "studenti_previsti":60},
        {"nome_lezione":"Programmazione ad Oggetti", "durata":3, "studenti_previsti":40},
        {"nome_lezione":"Intelligenza Artificiale", "durata":2, "studenti_previsti": 110},
        {"nome_lezione":"Algorimi e Strutture Dati", "durata":2, "studenti_previsti":100}
        ]

aule = [
        {"nome_aula":"Aula C1.10", "capienza":200},
        {"nome_aula":"Aula A1.7", "capienza":100},
        {"nome_aula":"Aula A1.2", "capienza":40}
    ]

solution = resolver(lezioni=lezioni, aule=aule, timeslot=timeslots)
solution.__reversed__()
print(solution)