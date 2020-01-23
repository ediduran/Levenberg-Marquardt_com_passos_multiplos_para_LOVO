using LinearAlgebra

# ----------------------------------------------------------------------------------------- #

# Exponencial

"""
Exponencial

Objetivo: Fornecer o modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*exp(x2*t)+x3.
"""
function exp!(x, t)
    return x[1] * exp(x[2] * t + x[3]) + x[4]
end

"""
Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna um vetor, com as derivadas parciais do modelo.
"""
function jacexp!(x, t)
    return [exp(x[2] * t + x[3]), x[1] * t * exp(x[2] * t + x[3]), x[1] * exp(x[2] * t + x[3]), 1.0]
end

# ----------------------------------------------------------------------------------------- #

# Logístico

"""
Logístico

Objetivo: Fornecer o modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1/(1+exp(x2*t+x3)).
"""
function log!(x, t)
    return x[1] / (1.0 + exp(x[2] * t + x[3]))
end

"""
Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna um vetor, com as derivadas parciais do modelo.
"""
function jaclog!(x, t)
    return [1.0 / (1.0 + exp(x[2] * t + x[3])), - t * x[1] * exp(x[2] * t + x[3]) / ((1.0 + exp(x[2] * t + x[3]))^(2.0)),
    - x[1] * exp(x[2] * t + x[3]) / ((1.0 + exp(x[2] * t + x[3]))^(2.0))]
end

# ----------------------------------------------------------------------------------------- #

# Polinomial de Grau 1

"""
Polinomial de Grau 1

Objetivo: Fornecer o modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*t+x2.
"""
function pol01!(x, t)
    return x[1] * t + x[2] 
end

"""
Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna um vetor, com as derivadas parciais do modelo.
"""
function jacpol01!(x, t)
    return [t, 1.0]
end

# ----------------------------------------------------------------------------------------- #

# Polinomial de Grau 3

"""
Polinomial de Grau 3
Objetivo: Fornecer o modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*t^3+x2*t^2+x[3]*t+x[4].
"""
function pol03!(x, t)
    return x[1] * t^(3.0) + x[2] * t^(2.0) + x[3] * t + x[4]
end

"""
Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna um vetor, com as derivadas parciais do modelo.
"""
function jacpol03!(x, t)
    return [t^(3.0), t^(2.0), t, 1.0]
end

# ----------------------------------------------------------------------------------------- #

# Senoidal(Tipo 1)

"""
Senoidal(Tipo 1)

Objetivo: Fornecer o modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*sin(x2*t+x3)+x4.
"""
function sen01!(x, t)
    return x[1] * sin(x[2] * t + x[3]) + x[4]    
end

"""
Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna um vetor, com as derivadas parciais do modelo.
"""
function jacsen01!(x, t)
    return [sin(x[2] * t + x[3]), x[1] * t * cos(x[2] * t + x[3]), x[1] * cos(x[2] * t + x[3]), 1.0]
end

# ----------------------------------------------------------------------------------------- #

# Senoidal(Tipo 2)

"""
Senoidal(Tipo 2)

Objetivo: Fornecer o modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*sin(x2*t)+x3*cos(x4*t)+x5.
"""
function sen02!(x, t)
    return x[1] * sin(x[2] * t) + x[3] * cos(x[4] * t) + x[5]
end

"""
Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.
Entrada:
x - vetor contendo os parâmetros a serem ajustados.
t - variável.
Saída:
Retorna um vetor, com as derivadas parciais do modelo.
"""
function jacsen02!(x, t)
    return [sin(x[2] * t), t * x[1] * cos(x[2] * t), cos(x[4] * t), -t * x[3] * sin(x[4] * t), 1.0]
end

# ----------------------------------------------------------------------------------------- #

