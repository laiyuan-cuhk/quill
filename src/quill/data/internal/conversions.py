from ..agda.syntax import (AgdaTerm, PiTerm, LamTerm, AppTerm, Reference, DeBruijn, LitTerm,
                           SortTerm, LevelTerm, UnsolvedMeta,
                           AgdaDefinition, ADT, Constructor, Record, Function, FunctionClause,
                           Postulate, Primitive,
                           Hole)
from .tree import Binary, Terminal, TreeBase
from enum import Enum, unique
from typing import NoReturn


@unique
class BinaryOp(Enum):
    PiSimple = 0
    PiDependent = 1
    Lambda = 2
    Application = 3


@unique
class NullaryOps(Enum):
    Sort = 0
    Level = 1
    Literal = 2
    Abs = 3


AgdaLeaf = NullaryOps | Reference | DeBruijn
AgdaOp = BinaryOp
AgdaTree = TreeBase[tuple[int, AgdaLeaf], tuple[int, AgdaOp]]


def term_to_ast(term: AgdaTerm, n: int = 1, bindings: tuple[int, ...] = ()) -> AgdaTree:
    from itertools import count

    counter = count(n)

    def go(t: AgdaTerm, _bindings: tuple[int, ...]) -> AgdaTree:
        idx = next(counter)
        match t:
            case PiTerm(domain, codomain, None):
                return Binary(
                    op=(idx, BinaryOp.PiSimple),
                    left=go(domain, _bindings),
                    right=go(codomain, _bindings))
            case PiTerm(domain, codomain, _):
                new_binding = next(counter)
                left = go(domain, _bindings)
                right = go(codomain, (new_binding, *(_bindings)))
                return Binary(op=(idx, BinaryOp.PiDependent), left=left, right=right)
            case LamTerm(_, body):
                abs_idx = next(counter)
                left = Terminal((abs_idx, NullaryOps.Abs))
                right = go(body, (abs_idx, *(_bindings)))
                return Binary(op=(idx, BinaryOp.Lambda), left=left, right=right)
            case AppTerm(head, argument):
                left = go(head, _bindings)
                right = go(argument, _bindings)
                return Binary(op=(idx, BinaryOp.Application), left=left, right=right)
            case Reference(_):
                return Terminal((idx, t))
            case DeBruijn(index):
                return Terminal((idx, DeBruijn(_bindings[index])))
            case SortTerm(_):
                return Terminal((idx, NullaryOps.Sort))
            case LitTerm(_):
                return Terminal((idx, NullaryOps.Literal))
            case LevelTerm(_):
                return Terminal((idx, NullaryOps.Level))
            case _:
                raise ValueError

    return go(term, bindings)


def definition_to_ast(definition: AgdaDefinition) -> NoReturn:
    match definition:
        case ADT(variants):
            raise NotImplementedError
        case Constructor(reference, variant):
            raise NotImplementedError
        case Record(fields, telescope):
            raise NotImplementedError
        case Function(clauses):
            raise NotImplementedError
        case Postulate():
            raise NotImplementedError
        case Primitive():
            raise NotImplementedError
        case _:
            raise ValueError


def clause_to_ast(clause: FunctionClause) -> NoReturn:
    raise NotImplementedError

