:- use_module(library(jpl)).

:- jpl_new('java.awt.Frame', ['CognitiveArchitecture'], Frame).

:- jpl_call(Frame, setSize, [800, 600], _).
:- jpl_call(Frame, setTitle, ['Cognitive Architecture'], _).
:- jpl_call(Frame, setVisible, [true], _).

:- jpl_new('java.awt.Label', ['Hello, World!'], Label).
:- jpl_call(Frame, add, [Label], _).

:- jpl_call(Frame, addMouseListener, [MouseListener], _).

MouseListener :-
    jpl_call(MouseEvent, getButton, [1], _),
    jpl_call(MouseEvent, getX, [X], _),
    jpl_call(MouseEvent, getY, [Y], _),
    format('Mouse clicked at (~d, ~d)~n', [X, Y]).

:- jpl_call(Frame, setDefaultCloseOperation, [3], _).
