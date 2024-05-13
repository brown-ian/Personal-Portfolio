import { describe, expect, test } from '@jest/globals'
import * as L from '../src/lang'
import * as Sexp from '../src/sexp'
import * as Trans from '../src/translator'
import * as TC from '../src/typechecker'
import * as Interp from '../src/interpreter'
import * as Runtime from '../src/runtime'
import * as PythonInterp from '../src/pythoninterpreter'

function compile (src: string, typecheck: boolean = true): L.Prog {
  const prog = Trans.translateProg(Sexp.parse(src))
  if (typecheck) {
    TC.checkWF(Runtime.initialCtx, prog)
  }
  return prog
}

function compileAndPrint (src: string, typecheck: boolean = true): string {
  return L.prettyProg(compile(src, typecheck))
}

function compileAndInterpret (src: string, typecheck: boolean = true): Interp.Output {
  return Interp.execute(Runtime.makeInitialEnv(), compile(src, typecheck))
}

const prog1 = `
  (define x 1)
  (define y 1)
  (print (+ x y))
  (assign x 10)
  (print (+ x y))
`
const pythonprog = `
  (define x 1)
  (define y 1)
  (assign x 10)
`

const prog2 = `
  (define result 0)
  (define factorial
    (lambda n Nat
      (if (zero? n)
          1
          (* n (factorial (- n 1))))))
  (assign result (factorial 5))
  (print result)
`
const pythonprog2 = `(define factorial
  (lambda n Nat
    (if (zero? n)
        1
        (* n (factorial (- n 1))))))`

const prog3 = 
  `(define record (rec x 2 y 3 z 4))
  (print (field record x))
  `
// testing the compiler
  const translate = PythonInterp.SchemetoPythonAssignments(pythonprog)
  console.log(translate)

  const translate2 = PythonInterp.SchemetoPythonAssignments(prog1)
console.log(translate2)

const translate4 = PythonInterp.SchemeToPythonRecords(prog3)
console.log(translate4)

const translate3 = PythonInterp.SchemetoPythonFunctions(pythonprog2)
console.log(translate3)



describe('interpretation', () => {
  test('prog1', () => {
    expect(compileAndInterpret(prog1, true)).toStrictEqual(['2', '11'])
  })
  test('prog2', () => {
    expect(compileAndInterpret(prog2, false)).toStrictEqual(['120'])
  })
  test('prog3', () => {
    expect(compileAndInterpret(prog3, false)).toStrictEqual(['2'])
  })
})

