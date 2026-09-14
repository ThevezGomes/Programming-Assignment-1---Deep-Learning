# AI LOG

Devido a complexidade do trabalho, foi necessário o uso de modelos de linguagem (LLM's) para auxiliar no desenvolvimento de certas etapas do projeto. 

Modelos utilizados:

- Gemini 3.7 Flash
- Gemini 3.1 Pro
- Claude Sonnet 4.6
- Claude Opus 4.6

Os principais problemas e dúvidas resolvidas com o auxílio dessas ferramentas foram:

- Auxílio na implementação de funções ao longo do projeto;
- Explicações de conceitos e ferramentas da área de Aprendizado Profundo;
- Busca por bugs e problemas no código.

Exemplos específicos:

- Após finalizarmos a parte 2 do projeto, percebemos que o modelo não apresentava um desempenho satisfatório.
  Por isso, recorremos ao uso de IA para analisar o código e procurar soluções para o problema.
  Após algumas análises, deduzimos que retreinar a parte do encoder pré-treinada poderia estar causando problemas de desempenho,
  por isso, experimentamos dividir o treinamento em duas partes: na primeira não treinamos os pesos da rede pré-treinada e na segunda treinamos o modelo por completo.
  Essa solução realmente melhorou um pouco o desempenho do modelo.

- Na parte de carregar os dados, precisávamos que todas as imagens possuíssem as mesmas dimensões, por isso, precisávamos redimensionar todas as imagens.
  Nesse processo, algumas das máscaras eram perdidas, por conta do redimensionamento e da resolução. Por conta disso, recorremos ao uso de IA para encontrar uma forma de recuperar as imagens perdidas.

Vale ressaltar que apesar do uso de LLMs, a estrutura do projeto, decisões, análises e as demais características da arquitetura do trabalho foram realizadas pelos integrantes do grupo.
