/* eslint-disable no-const-assign */
/* eslint-disable @typescript-eslint/restrict-plus-operands */
// Functions designed to designate assginments such as variables and other from scheme to python.
export function SchemetoPythonAssignments (code: string): string {
  const statements: string[] = code.split('\n').filter((statement) => statement.trim() !== '')
  let translatedCode = ''

  for (let j = 0; j < statements.length; j++) {
    const statement = statements[j].replace(/\(/g, '').replace(/\)/g, '').trim()
    const functions: string[] = statement.split(' ')

    if (functions[0] === 'define') {
      const variable: string = functions[1]
      const value: string = functions[2]
      translatedCode += `${variable} = ${value}\n`
    } else if (functions[0] === 'print') {
      const exp: string = functions.slice(1).join(' ')
      translatedCode += `print(${exp})\n`
    } else if (functions[0] === 'assign') {
      const tvar: string = functions[1]
      const value: number = parseInt(functions[2])
      translatedCode += `${tvar} = ${value}\n`
    } else {
      throw Error('did not enter if statements')
    }
  }

  return translatedCode
}

// INCOMPLETE
// Function design to translate functions from scheme to python. This does not yet work unfortunately.
export function SchemetoPythonFunctions (code: string): string {
  const lines: string[] = code.split('\n').filter((statement) => statement.trim() !== '')
  // eslint-disable-next-line prefer-const
  let translatedCode = ''

  for (let i = 0; i < lines.length; i++) {
    let translatedCode = lines[i].trim()
    if (translatedCode.startsWith('(define')) {
      const functionName = getFunctionName(translatedCode)
      const lambdaBody = getBody(translatedCode)
      translatedCode += `def ${functionName}(${getLambdaArgs(translatedCode)}):\n`
      translatedCode += `    ${SchemetoPythonFunctions(lambdaBody)}\n`
    } else if (translatedCode.startsWith('(lambda')) {
      const lambdaArgs = getLambdaArgs(translatedCode)
      const lambdaBody = getBody(translatedCode)
      translatedCode += `lambda ${lambdaArgs}: ${SchemetoPythonFunctions(lambdaBody)}`
    } else if (translatedCode.startsWith('(if')) {
      const condition = getIfCondition(translatedCode)
      const trueBranch = getIfTrueBranch(translatedCode)
      const falseBranch = getIfFalseBranch(translatedCode)
      translatedCode += `(${SchemetoPythonFunctions(trueBranch)}) if ${SchemetoPythonFunctions(condition)} else (${SchemetoPythonFunctions(falseBranch)})`
    } else if (translatedCode.startsWith('(*')) {
      const factors = getMathFactors(translatedCode)
      translatedCode += `(${factors.map(SchemetoPythonFunctions).join(' * ')})`
    } else if (translatedCode.startsWith('(-')) {
      const operands = getMathFactors(translatedCode)
      translatedCode += `(${operands.map(SchemetoPythonFunctions).join(' - ')})`
    } else {
      translatedCode += translatedCode
    }
  }

  return translatedCode
}
// aquires the fuction name from the given string/line
function getFunctionName (src: string): string {
  const parts = src.split(' ')
  return parts[1]
}
// aquires teh arguements that follow lambda in scheme language
function getLambdaArgs (src: string): string {
  const openingParenIndex = src.indexOf('(')
  const closingParenIndex = src.lastIndexOf(')')
  return src.slice(openingParenIndex + 1, closingParenIndex).trim()
}
// aquires the actual body from of the function
function getBody (src: string): string {
  const openingParenIndex = src.indexOf('(')
  const closingParenIndex = src.lastIndexOf(')')
  return src.slice(openingParenIndex + 1, closingParenIndex).trim()
}
// aquires the if statement if there is one.
function getIfCondition (src: string): string {
  const openingParenIndex = src.indexOf('(')
  const closingParenIndex = src.lastIndexOf(')')
  return src.slice(openingParenIndex + 1, closingParenIndex).trim()
}
// aquires the true if statement if there is one.
function getIfTrueBranch (src: string): string {
  const parts = src.split(/\(\s*if\s*|\s*else\s*\(/)
  return parts[1]
}
// aquires the false if statment if the true case is not avaiable.
function getIfFalseBranch (src: string): string {
  const parts = src.split(/\(\s*if\s*|\s*else\s*\(/)
  return parts[2]
}
// identifies mathematical statements
function getMathFactors (src: string): string[] {
  const openingParenIndex = src.indexOf('(')
  const closingParenIndex = src.lastIndexOf(')')
  const factors = src.slice(openingParenIndex + 1, closingParenIndex).trim().split(' ')
  return factors.slice(1)
}

export function SchemeToPythonRecords (code: string): string {
  const functions = code.split('\n')
  let translatedCode = ''

  for (let i = 0; i < functions.length; i++) {
    const newLine = functions[i].trim()
    if (newLine.startsWith('(define record (rec')) {
      const recordFields = getRecord(newLine)
      translatedCode += 'class Record:\n'

      for (let i = 0; i < recordFields.length; i += 2) {
        const fieldName = recordFields[i]
        const fieldValue = recordFields[i + 1]
        translatedCode += `    ${fieldName} = ${fieldValue}\n`
      }
    } else if (newLine.startsWith('(print (field record')) {
      const fieldName = getField(newLine)
      translatedCode += `print(Record.${fieldName})\n`
    }
  }
  return translatedCode
}

function getRecord (src: string): string[] {
  const record = src.match(/\w+/g)
  return (record != null) ? record.slice(3) : []
}

function getField (src: string): string {
  const fieldName = src.match(/\w+/g)
  return (fieldName != null) ? fieldName[3] : ''
}
