/* eslint-disable @typescript-eslint/naming-convention */
import * as L from './lang'

/** The output of our programs: a list of strings that our program printed. */
export type Output = string[]

/** The output of the language to python program: The string to be returned in python translation */

/** @returns the value that expression `e` evaluates to. */
export function evaluate (env: L.Env, e: L.Exp): L.Value {
  switch (e.tag) {
    case 'var': {
      if (env.has(e.value)) {
        return env.get(e.value)
      } else {
        throw new Error(`Runtime error: unbound variable '${e.value}'`)
      }
    }
    case 'num':
      return e
    case 'bool':
      return e
    case 'lam':
      return L.closure(e.param, e.body, env)
    case 'app': {
      const head = evaluate(env, e.head)
      const args = e.args.map(arg => evaluate(env, arg))
      if (head.tag === 'closure') {
        if (args.length !== 1) {
          throw new Error(`Runtime error: closure expects 1 argument but ${args.length} were given`)
        } else {
          return evaluate(head.env.extend1(head.param, args[0]), head.body)
        }
      } else if (head.tag === 'prim') {
        return head.fn(args)
      } else {
        throw new Error(`Runtime error: expected closure or primitive, but found '${L.prettyValue(head)}'`)
      }
    }
    case 'if': {
      const v = evaluate(env, e.e1)
      if (v.tag === 'bool') {
        return v.value ? evaluate(env, e.e2) : evaluate(env, e.e3)
      } else {
        throw new Error(`Type error: 'if' expects a boolean in guard position but a ${v.tag} was given.`)
      }
    }
    case 'field': {
      const v = evaluate(env, e.e)

      if (v.tag === 'record') {
        const val = v.map.get(e.f)
        if (val === undefined) {
          throw new Error(`Type error: the field ${e.f} is not inside of the record`)
        } else {
          return val
        }
      } else {
        throw new Error(`Type error: 'if' expects a rec in primary position but a ${v.tag} was given.`)
      }
    }
    case 'rec': {
      const record_values = new Map<string, L.Value>()
      for (const key of e.map.keys()) {
        const val = e.map.get(key)
        if (val === undefined) {
          throw new Error(`Type error: the field for ${key} is not inside of the record`)
        } else {
          const evaluated = evaluate(env, val)
          record_values.set(key, evaluated)
        }
      }
      return L.record(record_values)
    }
  }
}

/** @returns the result of executing program `prog` under environment `env` */
export function execute (env: L.Env, prog: L.Prog): Output {
  const output: Output = []
  for (const s of prog) {
    switch (s.tag) {
      case 'define': {
        const v = evaluate(env, s.exp)
        env.set(s.id, v)
        break
      }
      case 'assign': {
        const rhs = evaluate(env, s.exp)
        if (s.loc.tag === 'var') {
          if (env.has(s.loc.value)) {
            env.update(s.loc.value, rhs)
          } else {
            throw new Error(`Runtime error: unbound variable: ${s.loc.value}`)
          }
        } else {
          throw new Error(`Runtime error: cannot assign to non-location '${L.prettyExp(s.loc)}'}`)
        }
        break
      }
      case 'print': {
        const v = evaluate(env, s.exp)
        output.push(L.prettyValue(v))
        break
      }
    }
  }
  return output
}
