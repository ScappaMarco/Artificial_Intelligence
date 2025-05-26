# AI Assignment 2024/2025
### CSP Assignment Specification
Implementare il seguente problema adottando il formalismo dei Constraint Satisfaction Problems (CSPs) e utilizzando il package Python 'python-constraint' (https://github.com/python-constraint/python-constraint).
Il problema da implementare consiste nell'assegnare aule e timeslot a lezioni universitarie *IN UN DATO GIORNO* (solo per un unico giorno).
In particolare, dati:
 - un insieme di lezioni universitarie, ognuna caratterizzata dalla loro durata (in numero di ore) e dal numero di studenti previsti;
 - un insieme di aule, ognuna caratterizzata dal numero di posti a sedere;
 - un insieme di timeslot di disponibilità di ogni aula; ogni timeslot è della durata di 1 ora (e.g., si possono ipotizzare 9 timeslot in un giorno, dalle 8:30 alle 17:30, con un'ora di pausa pranzo);
Assegnare un'aula e uno o più timeslot a ogni lezione rispettando il vincolo sulla durata della lezione (i.e., se la lezione dura X ore, bisogna assegnare alla lezione X slot temporali *CONTIGUI* della stessa aula) e il vincolo sul numero di studenti vs. posti a sedere (i.e., se la lezione ha Y studenti attesi, possono essere assegnate a essa solo aule con numero di posti >= Y).

### Machine Learning Assignment Specification
Implementare in PyTorch:
 - Un'architettura di neural network per il task di handwritten digit recognition sul dataset MNIST.
 - Fase di validation dove si effettua il tuning degli iperparametri della suddetta architettura.
 - Fase di training (addestramento) del modello sottostante la suddetta architettura.
 - Fase di test del modello addestrato.
