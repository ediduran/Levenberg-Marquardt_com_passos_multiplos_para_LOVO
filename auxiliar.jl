using LinearAlgebra

# ----------------------------------------------------------------------------------#
"""
Entrada:
modelo -  modelo a ser ajustado.
deparcial - derivadas parciais do modelo.
x - vetor com os parâmetros iniciais.
t - dados de tempo.
y - dados observados.
r - número total de amonstras.
p - número de amostras confiáveis.
Saída: 
jacobiana - A matriz Jacobiana na restrição Ci
vaux[1:p] - A função F na restrição Ci
somaquad - Soma dos quadrados dos menores resíduos
indices - índices que compoe Ci 
"""
function resjac(modelo, deparcial, x, t, y, r, p)

    n = length(x)
    vaux = zeros(r)
    vaux2 = zeros(r)
    jacobiana = zeros(p, n)
    somaquad = 0.0

    for i=1:r
        vaux[i] = modelo(x, t[i]) - y[i]
    end
    vaux2 .= vaux.^(2.0)

    indices = Array{Int64}(1:1:r)
    for i = 1:p
        pmin = i
        for j = (pmin + 1):r
            if vaux2[j] < vaux2[pmin]
                pmin = j
            end
        end
        temp1 = vaux[i]
        temp2 = vaux2[i]
        temp3 = indices[i]
        vaux[i] = vaux[pmin]
        vaux2[i] = vaux2[pmin]
        indices[i] = indices[pmin]
        vaux[pmin] = temp1
        vaux2[pmin] = temp2
        indices[pmin] = temp3
    end

    somaquad = sum(vaux2[1:p])

    for i = 1:p
        jacobiana[i,:] = deparcial(x, t[indices[i]])
    end

    return jacobiana, vaux[1:p], somaquad, indices
end

# ----------------------------------------------------------------------------------#

"""
Objetivo: Calcula o valor da função objetivo.
Entrada:
modelo -  modelo a ser ajustado.
x - vetor com os parâmetros iniciais.
t - dados de tempo.
y - dados observados.
p - número de amostras confiáveis.
ind - indices que formam Ci.
Saída:
a construção da Fci
"""
function modci(modelo, x, t, y, p, ind)
    vaux = zeros(p)
    for i=1:p
        vaux[i] = modelo(x, t[ind[i]]) - y[ind[i]]
    end
    return vaux
    display(vaux)
end

# ----------------------------------------------------------------------------------#

"""Objetivo: Calcular o parâmetro LM, por Schwertner.
Entrada:
funcao - valor da função objetivo.
iteração - iteração k
Saída:
Retorna o parâmetro LM, para o método LOVO LM.
"""
function pd7(funcao, iteracao)
    return 2.0*funcao^(0.5)/(3.0*(iteracao + 1.0))
end


# ----------------------------------------------------------------------------------#