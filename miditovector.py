import torch

import numpy as np
import pretty_midi as pm

beatspersecond = 32


# returns a list of beat time steps
def beatList(endTime):
    timeperbeat = 1 / beatspersecond
    beats = []
    for i in range(int(endTime) * beatspersecond + beatspersecond):
        beats.append(i * timeperbeat)
    return beats


def songToVectors(filemame):
    songfile = pm.PrettyMIDI(filemame)
    #  puts song notes in list
    noteAsList = []
    for note in songfile.instruments[0].notes:  # [ Note Start, Note Duration, pitch, velocity]
        new_note = [note.start, note.end, note.pitch]
        # print(new_note)
        noteAsList.append(new_note)

    # Turns song into list of bitmaps
    notesVectors = []
    for time in beatList(songfile.get_end_time()): # Illiterate through the time of the song
        for note in noteAsList:
            if note[0] <= time <= note[1]:
                noteVector = np.zeros(129)
                noteVector[note[2]] = 1
                notesVectors.append(noteVector)

        endTimeVector = np.zeros(129)
        endTimeVector[128] = 1
        notesVectors.append(endTimeVector)
    return np.vstack(notesVectors)


def removeRests_BitMap(tensor):
    start = 0
    end = len(tensor) - 1


    if isinstance(tensor, torch.Tensor):
        referenceTensor = torch.zeros(129)
        referenceTensor[128] = 1
        referenceTensor = referenceTensor.type(tensor.type())
        for i in range(len(tensor)):
            start = i
            if not tensor[i].equal(referenceTensor):
                break
        for i in range(len(tensor) - 1, -1, -1):
            if tensor[i].equal(referenceTensor):
                end = i
            else:
                break
    elif isinstance(tensor, np.ndarray):
        referenceTensor = np.zeros(129)
        referenceTensor[128] = 1
        for i in range(len(tensor)):
            start = i
            if not np.array_equal(tensor[i], referenceTensor):
                break
        for i in range(len(tensor) - 1, -1, -1):
            if np.array_equal(tensor[i], referenceTensor):
                end = i
            else:

                break
    return tensor[start:end]



def tensorToNoteIndex(tensor):
    notes = []
    for note in tensor:
        bitIndex = np.where(note == 1)[0][0]
        notes.append(bitIndex)
    return notes


def noteIndexToTensor(notes):
    arraylist = []
    for note in notes:
        noteVector = np.zeros(129)
        noteVector[int(note)] = 1
        arraylist.append(noteVector)
    return np.vstack(arraylist)


def noteTensorToSong(notesVectors, name, dir="GeneratedMusic"):
    reinventedNotes = []
    time = 0
    for note in notesVectors:
        bitIndex = np.where(note == 1)[0][0]
        if bitIndex == 128:
            time += beatspersecond ** -1

        else:
            reinventedNotes.append([bitIndex, time, time + beatspersecond ** -1])

    notes_ready = []
    sorted(reinventedNotes, key=lambda x: x[0])

    for note in range(128):
        listNotes = sorted(list(filter(lambda x: x[0] == note, reinventedNotes)), key=lambda y: y[1])
        if len(listNotes) != 0:
            startTime = listNotes[0][1]
            mixedNotes = []
            for n in range(1, len(listNotes) - 1):
                if listNotes[n-1][2] != listNotes[n][1]:
                    mixedNotes.append([note, startTime, listNotes[n-1][2]])
                    startTime = listNotes[n][1]

            notes_ready += mixedNotes

    notes_ready = sorted(notes_ready, key=lambda x: x[1])

    newfile = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)
    for note in notes_ready:
        instrument.notes.append(pm.Note(velocity=100, pitch=note[0], start=note[1], end=note[2]))

    newfile.instruments.append(instrument)
    newfile.write("{}\\".format(dir) + name)


def noteIndexesToSong(notes, name, dir="GeneratedMusic"):
    reinventedNotes = []
    time = 0
    for note in notes:
        if note == 128:
            time += beatspersecond ** -1

        else:
            reinventedNotes.append([note, time, time + beatspersecond ** -1])

    notes_ready = []
    sorted(reinventedNotes, key=lambda x: x[0])

    for note in range(128):
        listNotes = sorted(list(filter(lambda x: x[0] == note, reinventedNotes)), key=lambda y: y[1])
        if len(listNotes) != 0:
            startTime = listNotes[0][1]
            mixedNotes = []
            for n in range(1, len(listNotes) - 1):
                if listNotes[n-1][2] != listNotes[n][1]:
                    mixedNotes.append([note, startTime, listNotes[n-1][2]])
                    startTime = listNotes[n][1]

            notes_ready += mixedNotes

    notes_ready = sorted(notes_ready, key=lambda x: x[1])

    newfile = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)
    for note in notes_ready:
        instrument.notes.append(pm.Note(velocity=100, pitch=note[0], start=note[1], end=note[2]))

    newfile.instruments.append(instrument)
    newfile.write("{}\\".format(dir) + name)

