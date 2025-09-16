# O que torna um sistema de IA um Agente?

> **Nota de Tradução**: Este conteúdo foi traduzido do original em inglês
> ["What makes an AI system an Agent?"](../en/intro/what-makes-an-agent.md) de "Agentic Design Patterns"
> por Antonio Gulli. Em caso de dúvidas, consulte a versão original.

Em termos simples, um agente de IA é um sistema projetado para perceber seu ambiente e tomar ações para alcançar um objetivo específico. É uma evolução de um Large Language Model (LLM) padrão, aprimorado com as habilidades de planejar, usar ferramentas e interagir com seus arredores. Pense em uma IA Agêntica como um assistente inteligente que aprende no trabalho. Ela segue um loop simples de cinco passos para realizar tarefas (veja Fig.1):

1. **Receber a Missão**: Você dá um objetivo, como "organize minha agenda."
2. **Escanear o Cenário**: Ela coleta todas as informações necessárias—lendo emails, verificando calendários e acessando contatos—para entender o que está acontecendo.
3. **Refletir**: Ela elabora um plano de ação considerando a abordagem ideal para alcançar o objetivo.
4. **Agir**: Ela executa o plano enviando convites, agendando reuniões e atualizando seu calendário.
5. **Aprender e Melhorar**: Ela observa resultados bem-sucedidos e se adapta adequadamente. Por exemplo, se uma reunião é reagendada, o sistema aprende com este evento para melhorar seu desempenho futuro.

*Fig.1: IA Agêntica funciona como um assistente inteligente, aprendendo continuamente através da experiência. Ela opera via um loop direto de cinco passos para realizar tarefas.*

Agentes estão se tornando cada vez mais populares em um ritmo impressionante. De acordo com estudos recentes, a maioria das grandes empresas de TI está ativamente usando esses agentes, e um quinto delas começou apenas no ano passado. Os mercados financeiros também estão prestando atenção. Até o final de 2024, startups de agentes de IA haviam levantado mais de $2 bilhões, e o mercado foi avaliado em $5,2 bilhões. Espera-se que exploda para quase $200 bilhões em valor até 2034. Em resumo, todos os sinais apontam para agentes de IA desempenhando um papel massivo em nossa economia futura.

Em apenas dois anos, o paradigma da IA mudou dramaticamente, passando de automação simples para sistemas sofisticados e autônomos (veja Fig. 2). Inicialmente, workflows dependiam de prompts e gatilhos básicos para processar dados com LLMs. Isso evoluiu com Retrieval-Augmented Generation (RAG), que melhorou a confiabilidade ao fundamentar modelos em informações factuais. Então vimos o desenvolvimento de Agentes de IA individuais capazes de usar várias ferramentas. Hoje, estamos entrando na era da IA Agêntica, onde uma equipe de agentes especializados trabalha em conjunto para alcançar objetivos complexos, marcando um salto significativo no poder colaborativo da IA.

*Fig 2.: Transicionando de LLMs para RAG, então para RAG Agêntico, e finalmente para IA Agêntica.*

A intenção deste livro é discutir os padrões de design de como agentes especializados podem trabalhar em conjunto e colaborar para alcançar objetivos complexos, e você verá um paradigma de colaboração e interação em cada capítulo.

Antes de fazer isso, vamos examinar exemplos que abrangem a gama de complexidade de agentes (veja Fig. 3).

## Nível 0: O Motor de Raciocínio Central

Embora um LLM não seja um agente em si, ele pode servir como o núcleo de raciocínio de um sistema agêntico básico. Em uma configuração 'Nível 0', o LLM opera sem ferramentas, memória ou interação com ambiente, respondendo apenas baseado em seu conhecimento pré-treinado. Sua força está em aproveitar seus extensos dados de treinamento para explicar conceitos estabelecidos. O trade-off para este poderoso raciocínio interno é uma completa falta de consciência de eventos atuais. Por exemplo, seria incapaz de nomear o vencedor do Oscar 2025 de "Melhor Filme" se essa informação estiver fora de seu conhecimento pré-treinado.

## Nível 1: O Solucionador de Problemas Conectado

Neste nível, o LLM se torna um agente funcional ao se conectar e utilizar ferramentas externas. Sua resolução de problemas não está mais limitada ao seu conhecimento pré-treinado. Em vez disso, pode executar uma sequência de ações para reunir e processar informações de fontes como a internet (via busca) ou bancos de dados (via Retrieval Augmented Generation, ou RAG). Para informações detalhadas, consulte o Capítulo 14.

Por exemplo, para encontrar novos programas de TV, o agente reconhece a necessidade de informação atual, usa uma ferramenta de busca para encontrá-la, e então sintetiza os resultados. Crucialmente, também pode usar ferramentas especializadas para maior precisão, como chamar uma API financeira para obter o preço atual da ação AAPL. Esta habilidade de interagir com o mundo externo através de múltiplos passos é a capacidade central de um agente Nível 1.

## Nível 2: O Solucionador de Problemas Estratégico

Neste nível, as capacidades de um agente se expandem significativamente, abrangendo planejamento estratégico, assistência proativa e auto-aperfeiçoamento, com engenharia de prompts e engenharia de contexto como habilidades centrais habilitadoras.

Primeiro, o agente vai além do uso de uma única ferramenta para enfrentar problemas complexos e multi-partes através de resolução estratégica de problemas. Conforme executa uma sequência de ações, ativamente realiza engenharia de contexto: o processo estratégico de selecionar, empacotar e gerenciar as informações mais relevantes para cada passo. Por exemplo, para encontrar uma cafeteria entre duas localizações, primeiro usa uma ferramenta de mapeamento. Então engenharia esta saída, curando um contexto curto e focado—talvez apenas uma lista de nomes de ruas—para alimentar uma ferramenta de busca local, prevenindo sobrecarga cognitiva e garantindo que o segundo passo seja eficiente e preciso. Para alcançar máxima precisão de uma IA, ela deve receber um contexto curto, focado e poderoso. Engenharia de contexto é a disciplina que realiza isso ao estrategicamente selecionar, empacotar e gerenciar as informações mais críticas de todas as fontes disponíveis. Ela efetivamente cura a atenção limitada do modelo para prevenir sobrecarga e garantir desempenho de alta qualidade e eficiente em qualquer tarefa dada. Para informações detalhadas, consulte o Apêndice A.

Este nível leva à operação proativa e contínua. Um assistente de viagem conectado ao seu email demonstra isso ao fazer engenharia do contexto de um email verboso de confirmação de voo; ele seleciona apenas os detalhes chave (números de voo, datas, localizações) para empacotar para chamadas subsequentes de ferramentas para seu calendário e uma API de clima.

Em campos especializados como engenharia de software, o agente gerencia um workflow inteiro aplicando esta disciplina. Quando atribuído um relatório de bug, ele lê o relatório e acessa o codebase, então estrategicamente faz engenharia dessas grandes fontes de informação em um contexto potente e focado que lhe permite escrever, testar e submeter eficientemente o patch de código correto.

Finalmente, o agente alcança auto-aperfeiçoamento ao refinar seus próprios processos de engenharia de contexto. Quando pede feedback sobre como um prompt poderia ter sido melhorado, está aprendendo como melhor curar suas entradas iniciais. Isso lhe permite automaticamente melhorar como empacota informações para tarefas futuras, criando um poderoso loop de feedback automatizado que aumenta sua precisão e eficiência ao longo do tempo. Para informações detalhadas, consulte o Capítulo 17.

*Fig. 3: Várias instâncias demonstrando o espectro de complexidade de agentes.*

## Nível 3: A Ascensão de Sistemas Multi-Agente Colaborativos

No Nível 3, vemos uma mudança significativa de paradigma no desenvolvimento de IA, afastando-se da busca por um único super-agente todo-poderoso e em direção à ascensão de sistemas multi-agente sofisticados e colaborativos. Em essência, esta abordagem reconhece que desafios complexos são frequentemente melhor resolvidos não por um único generalista, mas por uma equipe de especialistas trabalhando em conjunto. Este modelo espelha diretamente a estrutura de uma organização humana, onde diferentes departamentos são atribuídos papéis específicos e colaboram para enfrentar objetivos multifacetados. A força coletiva de tal sistema reside nesta divisão de trabalho e a sinergia criada através do esforço coordenado. Para informações detalhadas, consulte o Capítulo 7.

Para dar vida a este conceito, considere o workflow intricado de lançar um novo produto. Em vez de um agente tentando lidar com todos os aspectos, um agente "Gerente de Projeto" poderia servir como o coordenador central. Este gerente orquestraria todo o processo delegando tarefas para outros agentes especializados: um agente "Pesquisa de Mercado" para coletar dados do consumidor, um agente "Design de Produto" para desenvolver conceitos, e um agente "Marketing" para criar materiais promocionais. A chave para seu sucesso seria a comunicação perfeita e compartilhamento de informações entre eles, garantindo que todos os esforços individuais se alinhem para alcançar o objetivo coletivo.

Embora esta visão de automação autônoma baseada em equipe já esteja sendo desenvolvida, é importante reconhecer os obstáculos atuais. A efetividade de tais sistemas multi-agente está presentemente limitada pelas limitações de raciocínio dos LLMs que estão usando. Além disso, sua habilidade de genuinamente aprender uns dos outros e melhorar como uma unidade coesa ainda está em seus estágios iniciais. Superar esses gargalos tecnológicos é o próximo passo crítico, e fazê-lo desbloqueará a promessa profunda deste nível: a habilidade de automatizar workflows de negócios inteiros do início ao fim.

## O Futuro dos Agentes: Top 5 Hipóteses

O desenvolvimento de agentes de IA está progredindo em um ritmo sem precedentes em domínios como automação de software, pesquisa científica e atendimento ao cliente, entre outros. Embora os sistemas atuais sejam impressionantes, eles são apenas o começo. A próxima onda de inovação provavelmente focará em tornar agentes mais confiáveis, colaborativos e profundamente integrados em nossas vidas. Aqui estão cinco hipóteses principais para o que vem a seguir (veja Fig. 4).

### Hipótese 1: A Emergência do Agente Generalista

A primeira hipótese é que agentes de IA evoluirão de especialistas estreitos para verdadeiros generalistas capazes de gerenciar objetivos complexos, ambíguos e de longo prazo com alta confiabilidade. Por exemplo, você poderia dar a um agente um prompt simples como, "Planeje o retiro da minha empresa para 30 pessoas em Lisboa no próximo trimestre." O agente então gerenciaria todo o projeto por semanas, lidando com tudo desde aprovações de orçamento e negociações de voo até seleção de local e criação de um itinerário detalhado baseado no feedback dos funcionários, tudo enquanto fornece atualizações regulares. Alcançar este nível de autonomia exigirá avanços fundamentais em raciocínio de IA, memória e confiabilidade quase perfeita. Uma abordagem alternativa, mas não mutuamente exclusiva, é a ascensão de Small Language Models (SLMs). Este conceito "tipo Lego" envolve compor sistemas de pequenos agentes especialistas em vez de escalar um único modelo monolítico. Este método promete sistemas que são mais baratos, mais rápidos para debugar e mais fáceis de implantar. Ultimamente, o desenvolvimento de grandes modelos generalistas e a composição de menores especializados são ambos caminhos plausíveis para frente, e eles poderiam até se complementar.

### Hipótese 2: Personalização Profunda e Descoberta Proativa de Objetivos

A segunda hipótese postula que agentes se tornarão parceiros profundamente personalizados e proativos. Estamos testemunhando a emergência de uma nova classe de agente: o parceiro proativo. Ao aprender de seus padrões e objetivos únicos, esses sistemas estão começando a mudar de apenas seguir ordens para antecipar suas necessidades. Sistemas de IA operam como agentes quando vão além de simplesmente responder a chats ou instruções. Eles iniciam e executam tarefas em nome do usuário, colaborando ativamente no processo. Isso vai além da simples execução de tarefas para o reino da descoberta proativa de objetivos.

Por exemplo, se você está explorando energia sustentável, o agente poderia identificar seu objetivo latente e proativamente apoiá-lo sugerindo cursos ou resumindo pesquisas. Embora esses sistemas ainda estejam se desenvolvendo, sua trajetória é clara. Eles se tornarão cada vez mais proativos, aprendendo a tomar iniciativa em seu nome quando altamente confiantes de que a ação será útil. Ultimamente, o agente se torna um aliado indispensável, ajudando você a descobrir e alcançar ambições que ainda não articulou completamente.

*Fig. 4: Cinco hipóteses sobre o futuro dos agentes*

### Hipótese 3: Incorporação e Interação com o Mundo Físico

Esta hipótese prevê agentes quebrando livre de seus confinamentos puramente digitais para operar no mundo físico. Ao integrar IA agêntica com robótica, veremos a ascensão de "agentes incorporados." Em vez de apenas reservar um técnico, você poderia pedir ao seu agente doméstico para consertar uma torneira que vaza. O agente usaria seus sensores de visão para perceber o problema, acessar uma biblioteca de conhecimento de encanamento para formular um plano, e então controlar seus manipuladores robóticos com precisão para realizar o reparo. Isso representaria um passo monumental, conectando a lacuna entre inteligência digital e ação física, e transformando tudo desde manufatura e logística até cuidados com idosos e manutenção doméstica.

### Hipótese 4: A Economia Dirigida por Agentes

A quarta hipótese é que agentes altamente autônomos se tornarão participantes ativos na economia, criando novos mercados e modelos de negócio. Poderíamos ver agentes atuando como entidades econômicas independentes, encarregados de maximizar um resultado específico, como lucro. Um empreendedor poderia lançar um agente para administrar um negócio de e-commerce inteiro. O agente identificaria produtos em tendência analisando mídias sociais, geraria cópias de marketing e visuais, gerenciaria logística da cadeia de suprimentos interagindo com outros sistemas automatizados, e ajustaria dinamicamente preços baseado em demanda em tempo real. Esta mudança criaria uma nova "economia de agentes" hiper-eficiente operando em uma velocidade e escala impossível para humanos gerenciarem diretamente.

### Hipótese 5: O Sistema Multi-Agente Metamórfico Dirigido por Objetivos

Esta hipótese postula a emergência de sistemas inteligentes que operam não de programação explícita, mas de um objetivo declarado. O usuário simplesmente declara o resultado desejado, e o sistema autonomamente descobre como alcançá-lo. Isso marca uma mudança fundamental em direção a sistemas multi-agente metamórficos capazes de verdadeiro auto-aperfeiçoamento tanto no nível individual quanto coletivo.

Este sistema seria uma entidade dinâmica, não um único agente. Teria a habilidade de analisar sua própria performance e modificar a topologia de sua força de trabalho multi-agente, criando, duplicando ou removendo agentes conforme necessário para formar a equipe mais efetiva para a tarefa em questão. Esta evolução acontece em múltiplos níveis:

* **Modificação Arquitetural**: No nível mais profundo, agentes individuais podem reescrever seu próprio código fonte e re-arquitetar suas estruturas internas para maior eficiência, como na hipótese original.
* **Modificação Instrucional**: Em um nível mais alto, o sistema continuamente realiza engenharia automática de prompts e engenharia de contexto. Ele refina as instruções e informações dadas a cada agente, garantindo que estejam operando com orientação ideal sem qualquer intervenção humana.

Por exemplo, um empreendedor simplesmente declararia a intenção: "Lance um negócio de e-commerce bem-sucedido vendendo café artesanal." O sistema, sem programação adicional, entraria em ação. Poderia inicialmente gerar um agente "Pesquisa de Mercado" e um agente "Branding". Baseado nos achados iniciais, poderia decidir remover o agente de branding e gerar três novos agentes especializados: um agente "Design de Logo", um agente "Plataforma de Webstore", e um agente "Cadeia de Suprimentos". Constantemente ajustaria seus prompts internos para melhor performance. Se o agente de webstore se tornar um gargalo, o sistema poderia duplicá-lo em três agentes paralelos para trabalhar em diferentes partes do site, efetivamente re-arquitetando sua própria estrutura instantaneamente para melhor alcançar o objetivo declarado.

## Conclusão

Em essência, um agente de IA representa um salto significativo dos modelos tradicionais, funcionando como um sistema autônomo que percebe, planeja e age para alcançar objetivos específicos. A evolução desta tecnologia está avançando de agentes únicos usando ferramentas para sistemas multi-agente complexos e colaborativos que enfrentam objetivos multifacetados. Hipóteses futuras predizem a emergência de agentes generalistas, personalizados e até fisicamente incorporados que se tornarão participantes ativos na economia. Este desenvolvimento contínuo sinaliza uma grande mudança de paradigma em direção a sistemas auto-aperfeiçoadores dirigidos por objetivos, prontos para automatizar workflows inteiros e fundamentalmente redefinir nossa relação com a tecnologia.

## Referências

1. Cloudera, Inc. (April 2025), 96% of enterprises are increasing their use of AI agents. https://www.cloudera.com/about/news-and-blogs/press-releases/2025-04-16-96-percent-of-enterprises-are-expanding-use-of-ai-agents-according-to-latest-data-from-cloudera.html
2. Autonomous generative AI agents: https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2025/autonomous-generative-ai-agents-still-under-development.html
3. Market.us. Global Agentic AI Market Size, Trends and Forecast 2025–2034. https://market.us/report/agentic-ai-market/

---

*De "Agentic Design Patterns" por Antonio Gulli*