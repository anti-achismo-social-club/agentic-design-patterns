# Capítulo 1: Encadeamento de Prompts

*Conteúdo original: 12 páginas - por Antonio Gulli*
*Tradução para PT-BR: Esta tradução visa tornar o conteúdo acessível para desenvolvedores brasileiros, mantendo a precisão técnica do material original.*

## Visão Geral do Padrão de Encadeamento de Prompts

O encadeamento de prompts, às vezes referido como padrão Pipeline, representa um paradigma poderoso para lidar com tarefas complexas ao utilizar modelos de linguagem grandes (LLMs). Em vez de esperar que um LLM resolva um problema complexo em uma única etapa monolítica, o encadeamento de prompts defende uma estratégia de dividir para conquistar. A ideia central é quebrar o problema original e intimidador em uma sequência de subproblemas menores e mais gerenciáveis. Cada subproblema é abordado individualmente através de um prompt especificamente projetado, e a saída gerada de um prompt é estrategicamente alimentada como entrada no prompt subsequente da cadeia.

Esta técnica de processamento sequencial introduz inerentemente modularidade e clareza na interação com LLMs. Ao decompor uma tarefa complexa, torna-se mais fácil entender e debugar cada etapa individual, tornando o processo geral mais robusto e interpretável. Cada etapa na cadeia pode ser meticulosamente elaborada e otimizada para focar em um aspecto específico do problema maior, levando a saídas mais precisas e focadas.

A saída de uma etapa atuando como entrada para a próxima é crucial. Esta passagem de informação estabelece uma cadeia de dependência, daí o nome, onde o contexto e os resultados de operações anteriores guiam o processamento subsequente. Isso permite que o LLM construa sobre seu trabalho anterior, refine sua compreensão e progrida gradualmente em direção à solução desejada.

Além disso, o encadeamento de prompts não é apenas sobre quebrar problemas; também possibilita a integração de conhecimento e ferramentas externas. Em cada etapa, o LLM pode ser instruído a interagir com sistemas externos, APIs ou bancos de dados, enriquecendo seu conhecimento e habilidades além de seus dados internos de treinamento. Esta capacidade expande dramaticamente o potencial dos LLMs, permitindo que funcionem não apenas como modelos isolados, mas como componentes integrais de sistemas mais amplos e inteligentes.

A significância do encadeamento de prompts se estende além da simples resolução de problemas. Serve como uma técnica fundamental para construir agentes de IA sofisticados. Estes agentes podem utilizar cadeias de prompts para planejar, raciocinar e agir autonomamente em ambientes dinâmicos. Ao estruturar estrategicamente a sequência de prompts, um agente pode se envolver em tarefas que requerem raciocínio multi-etapas, planejamento e tomada de decisão. Tais fluxos de trabalho de agentes podem imitar processos de pensamento humano mais de perto, permitindo interações mais naturais e eficazes com domínios e sistemas complexos.

### Limitações de Prompts Únicos

Para tarefas multifacetadas, usar um único prompt complexo para um LLM pode ser ineficiente, fazendo com que o modelo lute com restrições e instruções, potencialmente levando a:

- **Negligência de instruções**: onde partes do prompt são ignoradas
- **Deriva contextual**: onde o modelo perde o rastro do contexto inicial
- **Propagação de erros**: onde erros iniciais se amplificam
- **Problemas de janela de contexto**: onde o modelo recebe informação insuficiente para responder adequadamente
- **Alucinação**: onde a carga cognitiva aumenta a chance de informação incorreta

Por exemplo, uma consulta pedindo para analisar um relatório de pesquisa de mercado, resumir descobertas, identificar tendências com pontos de dados e rascunhar um email corre o risco de falhar, pois o modelo pode resumir bem, mas falhar em extrair dados ou rascunhar um email adequadamente.

### Confiabilidade Aprimorada Através da Decomposição Sequencial

O encadeamento de prompts aborda esses desafios ao quebrar a tarefa complexa em um fluxo de trabalho focado e sequencial, o que melhora significativamente a confiabilidade e controle. Dado o exemplo acima, uma abordagem pipeline ou encadeada pode ser descrita da seguinte forma:

1. **Prompt Inicial (Resumo)**: "Resuma as descobertas principais do seguinte relatório de pesquisa de mercado: [texto]." O foco único do modelo é o resumo, aumentando a precisão desta etapa inicial.

2. **Segundo Prompt (Identificação de Tendências)**: "Usando o resumo, identifique as três principais tendências emergentes e extraia os pontos de dados específicos que suportam cada tendência: [saída da etapa 1]." Este prompt agora é mais restrito e constrói diretamente sobre uma saída validada.

3. **Terceiro Prompt (Composição de Email)**: "Rascunhe um email conciso para a equipe de marketing que delineie as seguintes tendências e seus dados de suporte: [saída da etapa 2]."

Esta decomposição permite um controle mais granular sobre o processo. Cada etapa é mais simples e menos ambígua, o que reduz a carga cognitiva no modelo e leva a uma saída final mais precisa e confiável. Esta modularidade é análoga a um pipeline computacional onde cada função executa uma operação específica antes de passar seu resultado para a próxima.

### O Papel da Saída Estruturada

A confiabilidade de uma cadeia de prompts é altamente dependente da integridade dos dados passados entre etapas. Se a saída de um prompt for ambígua ou mal formatada, o prompt subsequente pode falhar devido à entrada defeituosa. Para mitigar isso, especificar um formato de saída estruturado, como JSON ou XML, é crucial.

Por exemplo, a saída da etapa de identificação de tendências poderia ser formatada como um objeto JSON:

```json
{
 "trends": [
   {
     "trend_name": "Personalização Baseada em IA",
     "supporting_data": "73% dos consumidores preferem fazer negócios com marcas que usam informações pessoais para tornar suas experiências de compra mais relevantes."
   },
   {
     "trend_name": "Marcas Sustentáveis e Éticas",
     "supporting_data": "Vendas de produtos com alegações relacionadas a ESG cresceram 28% nos últimos cinco anos, comparado a 20% para produtos sem essas alegações."
   }
 ]
}
```

Este formato estruturado garante que os dados sejam legíveis por máquina e possam ser precisamente analisados e inseridos no próximo prompt sem ambiguidade. Esta prática minimiza erros que podem surgir da interpretação de linguagem natural e é um componente chave na construção de sistemas robustos baseados em LLM de múltiplas etapas.

## Aplicações Práticas e Casos de Uso

O encadeamento de prompts é um padrão versátil aplicável em uma ampla gama de cenários ao construir sistemas agênticos. Sua utilidade central reside em quebrar problemas complexos em etapas sequenciais e gerenciáveis. Aqui estão várias aplicações práticas e casos de uso:

### 1. Fluxos de Trabalho de Processamento de Informação

Muitas tarefas envolvem processar informação bruta através de múltiplas transformações. Por exemplo, resumir um documento, extrair entidades-chave e então usar essas entidades para consultar um banco de dados ou gerar um relatório. Uma cadeia de prompts poderia ser:

- **Prompt 1**: Extrair conteúdo de texto de uma URL ou documento dado.
- **Prompt 2**: Resumir o texto limpo.
- **Prompt 3**: Extrair entidades específicas (ex: nomes, datas, localizações) do resumo ou texto original.
- **Prompt 4**: Usar as entidades para pesquisar uma base de conhecimento interna.
- **Prompt 5**: Gerar um relatório final incorporando o resumo, entidades e resultados de pesquisa.

### 2. Resposta a Consultas Complexas

Responder questões complexas que requerem múltiplas etapas de raciocínio ou recuperação de informação é um caso de uso primário. Por exemplo, "Quais foram as principais causas da quebra do mercado de ações em 1929, e como a política governamental respondeu?"

- **Prompt 1**: Identificar as sub-questões centrais na consulta do usuário (causas da quebra, resposta governamental).
- **Prompt 2**: Pesquisar ou recuperar informação especificamente sobre as causas da quebra de 1929.
- **Prompt 3**: Pesquisar ou recuperar informação especificamente sobre a resposta política do governo à quebra do mercado de ações de 1929.
- **Prompt 4**: Sintetizar a informação das etapas 2 e 3 em uma resposta coerente à consulta original.

### 3. Extração e Transformação de Dados

A conversão de texto não estruturado em um formato estruturado é tipicamente alcançada através de um processo iterativo:

- **Prompt 1**: Tentar extrair campos específicos (ex: nome, endereço, valor) de um documento de fatura.
- **Processamento**: Verificar se todos os campos obrigatórios foram extraídos e se atendem aos requisitos de formato.
- **Prompt 2 (Condicional)**: Se campos estão faltando ou mal formados, criar um novo prompt pedindo ao modelo para encontrar especificamente a informação faltante/mal formada.
- **Processamento**: Validar os resultados novamente. Repetir se necessário.
- **Saída**: Fornecer os dados estruturados extraídos e validados.

### 4. Fluxos de Trabalho de Geração de Conteúdo

A composição de conteúdo complexo é uma tarefa procedimental que é tipicamente decomposta em fases distintas:

- **Prompt 1**: Gerar 5 ideias de tópicos baseadas no interesse geral de um usuário.
- **Processamento**: Permitir que o usuário selecione uma ideia ou escolher automaticamente a melhor.
- **Prompt 2**: Baseado no tópico selecionado, gerar um esboço detalhado.
- **Prompt 3**: Escrever uma seção de rascunho baseada no primeiro ponto do esboço.
- **Prompt 4**: Escrever uma seção de rascunho baseada no segundo ponto do esboço, fornecendo a seção anterior para contexto.
- **Prompt 5**: Revisar e refinar o rascunho completo para coerência, tom e gramática.

### 5. Geração e Refinamento de Código

A geração de código funcional é tipicamente um processo de múltiplas etapas:

- **Prompt 1**: Entender a solicitação do usuário para uma função de código. Gerar pseudocódigo ou um esboço.
- **Prompt 2**: Escrever o rascunho inicial do código baseado no esboço.
- **Prompt 3**: Identificar potenciais erros ou áreas para melhoria no código.
- **Prompt 4**: Reescrever ou refinar o código baseado nos problemas identificados.
- **Prompt 5**: Adicionar documentação ou casos de teste.

## Exemplo de Código Prático

Implementar encadeamento de prompts varia de chamadas de função sequenciais diretas dentro de um script ao uso de frameworks especializados projetados para gerenciar fluxo de controle, estado e integração de componentes. O código a seguir implementa uma cadeia de prompts de duas etapas que funciona como um pipeline de processamento de dados:

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Inicializar o Modelo de Linguagem
llm = ChatOpenAI(temperature=0)

# --- Prompt 1: Extrair Informação ---
prompt_extract = ChatPromptTemplate.from_template(
   "Extraia as especificações técnicas do seguinte texto:\n\n{text_input}"
)

# --- Prompt 2: Transformar para JSON ---
prompt_transform = ChatPromptTemplate.from_template(
   "Transforme as seguintes especificações em um objeto JSON com 'cpu', 'memory' e 'storage' como chaves:\n\n{specifications}"
)

# --- Construir a Cadeia usando LCEL ---
extraction_chain = prompt_extract | llm | StrOutputParser()

# A cadeia completa passa a saída da cadeia de extração para a variável 'specifications'
# do prompt de transformação.
full_chain = (
   {"specifications": extraction_chain}
   | prompt_transform
   | llm
   | StrOutputParser()
)

# --- Executar a Cadeia ---
input_text = "O novo modelo de laptop possui um processador octa-core de 3.5 GHz, 16GB de RAM e um SSD NVMe de 1TB."

# Executar a cadeia com o dicionário de texto de entrada.
final_result = full_chain.invoke({"text_input": input_text})

print("\n--- Saída JSON Final ---")
print(final_result)
```

Este código demonstra como usar LangChain para processar texto através de dois prompts separados: um para extrair especificações técnicas e outro para formatar essas especificações em um objeto JSON.

## Engenharia de Contexto e Engenharia de Prompt

A Engenharia de Contexto é a disciplina sistemática de projetar, construir e entregar um ambiente informacional completo a um modelo de IA antes da geração de tokens. Esta metodologia afirma que a qualidade da saída de um modelo é menos dependente da arquitetura do modelo em si e mais da riqueza do contexto fornecido.

Representa uma evolução significativa da engenharia de prompt tradicional, que foca principalmente em otimizar a formulação da consulta imediata de um usuário. A Engenharia de Contexto expande este escopo para incluir várias camadas de informação, tais como:

- **Prompts de sistema**: Instruções fundamentais definindo os parâmetros operacionais da IA
- **Dados externos**: Documentos recuperados de bases de conhecimento
- **Saídas de ferramentas**: Resultados de chamadas de API externas para dados em tempo real
- **Dados implícitos**: Identidade do usuário, histórico de interação e estado ambiental

## Principais Conclusões

- **Encadeamento de Prompts** quebra tarefas complexas em uma sequência de etapas menores e focadas
- Cada etapa em uma cadeia envolve uma chamada de LLM ou lógica de processamento, usando a saída da etapa anterior como entrada
- Este padrão melhora a confiabilidade e gerenciabilidade de interações complexas com modelos de linguagem
- Frameworks como LangChain/LangGraph e Google ADK fornecem ferramentas robustas para definir, gerenciar e executar essas sequências de múltiplas etapas
- **Engenharia de Contexto** é crucial para construir ambientes informacionais abrangentes que permitem performance agêntica avançada

## Resumo Visual

*Padrão de Encadeamento de Prompts: Agentes recebem uma série de prompts do usuário, com a saída de cada agente servindo como entrada para o próximo na cadeia.*

## Quando Usar Este Padrão

**Use este padrão quando:**
- Uma tarefa é muito complexa para um único prompt
- A tarefa envolve múltiplas etapas de processamento distintas
- Você precisa de interação com ferramentas externas entre etapas
- Construindo sistemas agênticos que precisam realizar raciocínio multi-etapas e manter estado

## Conclusão

Ao desconstruir problemas complexos em uma sequência de sub-tarefas mais simples e gerenciáveis, o encadeamento de prompts fornece um framework robusto para guiar modelos de linguagem grandes. Esta estratégia de "dividir para conquistar" melhora significativamente a confiabilidade e controle da saída ao focar o modelo em uma operação específica por vez. Como um padrão fundamental, permite o desenvolvimento de agentes de IA sofisticados capazes de raciocínio multi-etapas, integração de ferramentas e gerenciamento de estado.

## Referências

1. LangChain Documentation on LCEL: https://python.langchain.com/v0.2/docs/core_modules/expression_language/
2. LangGraph Documentation: https://langchain-ai.github.io/langgraph/
3. Prompt Engineering Guide - Chaining Prompts: https://www.promptingguide.ai/techniques/chaining
4. OpenAI API Documentation: https://platform.openai.com/docs/guides/gpt/prompting
5. Crew AI Documentation: https://docs.crewai.com/
6. Google AI for Developers: https://cloud.google.com/discover/what-is-prompt-engineering?hl=en
7. Vertex Prompt Optimizer: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-optimizer
- Testes e validação podem ser realizados no nível de componente

## Implementação

### Estrutura Básica de Cadeia
```python
def prompt_chain(initial_input):
    # Etapa 1: Análise
    analysis = call_llm(analysis_prompt, initial_input)

    # Etapa 2: Planejamento
    plan = call_llm(planning_prompt, analysis)

    # Etapa 3: Execução
    result = call_llm(execution_prompt, plan)

    return result
```

### Gerenciamento Avançado de Cadeia
- Implementar tratamento de erros e lógica de retry
- Adicionar logging e monitoramento em cada etapa
- Incluir ramificação condicional baseada em resultados intermediários
- Armazenar estados intermediários para debugging e análise

## Exemplos de Código

### Exemplo 1: Cadeia de Análise de Documento
```python
class DocumentAnalysisChain:
    def __init__(self):
        self.summarizer_prompt = "Resuma o seguinte documento: {document}"
        self.extractor_prompt = "Extraia entidades-chave de: {summary}"
        self.classifier_prompt = "Classifique o tipo de documento baseado em: {entities}"

    def analyze(self, document):
        # Etapa 1: Resumir
        summary = self.llm_call(self.summarizer_prompt, document=document)

        # Etapa 2: Extrair entidades
        entities = self.llm_call(self.extractor_prompt, summary=summary)

        # Etapa 3: Classificar
        classification = self.llm_call(self.classifier_prompt, entities=entities)

        return {
            'summary': summary,
            'entities': entities,
            'classification': classification
        }
```

### Exemplo 2: Cadeia de Escrita Criativa
```python
def creative_writing_chain(topic, style):
    # Gerar esboço
    outline_prompt = f"Crie um esboço para uma história {style} sobre {topic}"
    outline = generate_response(outline_prompt)

    # Desenvolver personagens
    character_prompt = f"Baseado neste esboço: {outline}, crie personagens detalhados"
    characters = generate_response(character_prompt)

    # Escrever história
    story_prompt = f"Escreva uma história {style} usando esboço: {outline} e personagens: {characters}"
    story = generate_response(story_prompt)

    return story
```

## Melhores Práticas

### Princípios de Design
- **Responsabilidade Única**: Cada prompt deve ter um propósito claro
- **Interfaces Claras**: Definir formatos explícitos de entrada/saída entre prompts
- **Limites de Erro**: Implementar tratamento adequado de erros em cada etapa
- **Logging**: Manter logs abrangentes da execução da cadeia

### Estratégias de Otimização
- **Engenharia de Prompt**: Otimizar cada prompt individualmente para sua tarefa específica
- **Cache**: Cache resultados intermediários quando apropriado
- **Execução Paralela**: Identificar oportunidades para processamento paralelo
- **Gerenciamento de Recursos**: Monitorar uso de tokens e custos de API

### Teste e Validação
- Testar cada prompt isoladamente antes da integração
- Criar suítes de teste abrangentes para toda a cadeia
- Monitorar performance e precisão da cadeia ao longo do tempo
- Implementar testes A/B para variações de prompt

## Armadilhas Comuns

### Perda de Informação
- **Problema**: Contexto importante se perde entre etapas da cadeia
- **Solução**: Implementar gerenciamento adequado de estado e preservação de contexto
- **Mitigação**: Incluir contexto relevante em cada prompt

### Fragilidade da Cadeia
- **Problema**: Falha em uma etapa quebra toda a cadeia
- **Solução**: Implementar tratamento robusto de erros e mecanismos de fallback
- **Mitigação**: Projetar caminhos redundantes para operações críticas

### Sobre-Engenharia
- **Problema**: Criar cadeias desnecessariamente complexas para tarefas simples
- **Solução**: Começar simples e adicionar complexidade apenas quando necessário
- **Mitigação**: Revisão regular e simplificação da lógica da cadeia

### Acúmulo de Custos
- **Problema**: Múltiplas chamadas de API podem se tornar caras
- **Solução**: Otimizar eficiência de prompt e implementar cache
- **Mitigação**: Monitorar custos e implementar limites de uso

### Limitações da Janela de Contexto
- **Problema**: Contexto acumulado excede limites do modelo
- **Solução**: Implementar estratégias de poda e resumo de contexto
- **Mitigação**: Projetar cadeias com restrições de janela de contexto em mente

## Conclusão

O encadeamento de prompts fornece uma base robusta para construir sistemas de IA agênticos sofisticados. Ao quebrar tarefas complexas em etapas sequenciais gerenciáveis, este padrão permite que desenvolvedores criem fluxos de trabalho de IA mais confiáveis, mantíveis e interpretáveis. O sucesso com encadeamento de prompts requer atenção cuidadosa aos princípios de design, tratamento adequado de erros e otimização contínua de componentes individuais da cadeia.