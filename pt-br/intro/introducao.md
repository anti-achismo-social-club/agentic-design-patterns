# Introdução

> **Nota de Tradução**: Este conteúdo foi traduzido do original em inglês
> ["Introduction"](../en/intro/introduction.md) de "Agentic Design Patterns"
> por Antonio Gulli. Em caso de dúvidas, consulte a versão original.

Bem-vindos a "Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems." Ao observarmos o cenário da inteligência artificial moderna, vemos uma evolução clara de programas simples e reativos para entidades sofisticadas e autônomas capazes de compreender contexto, tomar decisões e interagir dinamicamente com seu ambiente e outros sistemas. Estes são os agentes inteligentes e os sistemas agênticos que eles compõem.

O advento de poderosos large language models (LLMs) forneceu capacidades sem precedentes para compreender e gerar conteúdo humanizado como texto e mídia, servindo como motor cognitivo para muitos desses agentes. No entanto, orquestrar essas capacidades em sistemas que podem alcançar objetivos complexos de forma confiável requer mais do que apenas um modelo poderoso. Requer estrutura, design e uma abordagem cuidadosa sobre como o agente percebe, planeja, age e interage.

Pense em construir sistemas inteligentes como criar uma obra complexa de arte ou engenharia em uma tela. Esta tela não é um espaço visual em branco, mas sim a infraestrutura e frameworks subjacentes que fornecem o ambiente e ferramentas para seus agentes existirem e operarem. É a fundação sobre a qual você construirá sua aplicação inteligente, gerenciando estado, comunicação, acesso a ferramentas e o fluxo de lógica.

Construir efetivamente nesta tela agêntica exige mais do que simplesmente juntar componentes. Requer compreender técnicas comprovadas – padrões – que abordam desafios comuns no design e implementação do comportamento de agentes. Assim como padrões arquiteturais guiam a construção de um edifício, ou design patterns estruturam software, padrões de design agênticos fornecem soluções reutilizáveis para os problemas recorrentes que você enfrentará ao dar vida a agentes inteligentes em sua tela escolhida.

## O que são Sistemas Agênticos?

Em sua essência, um sistema agêntico é uma entidade computacional projetada para perceber seu ambiente (tanto digital quanto potencialmente físico), tomar decisões informadas baseadas nessas percepções e um conjunto de objetivos predefinidos ou aprendidos, e executar ações para alcançar esses objetivos autonomamente. Diferentemente do software tradicional, que segue instruções rígidas, passo a passo, agentes exibem um grau de flexibilidade e iniciativa.

Imagine que você precisa de um sistema para gerenciar consultas de clientes. Um sistema tradicional pode seguir um script fixo. Um sistema agêntico, no entanto, poderia perceber as nuances da consulta de um cliente, acessar bases de conhecimento, interagir com outros sistemas internos (como gerenciamento de pedidos), potencialmente fazer perguntas esclarecedoras e resolver proativamente a questão, talvez até antecipando necessidades futuras. Esses agentes operam na tela da infraestrutura de sua aplicação, utilizando os serviços e dados disponíveis para eles.

Sistemas agênticos são frequentemente caracterizados por recursos como autonomia, permitindo que atuem sem supervisão humana constante; proatividade, iniciando ações em direção a seus objetivos; e reatividade, respondendo efetivamente a mudanças em seu ambiente. Eles são fundamentalmente orientados a objetivos, trabalhando constantemente em direção a metas. Uma capacidade crítica é o uso de ferramentas, permitindo que interajam com APIs externas, bancos de dados ou serviços – efetivamente alcançando além de sua tela imediata. Eles possuem memória, retêm informações através de interações, e podem se engajar em comunicação com usuários, outros sistemas ou mesmo outros agentes operando na mesma tela ou telas conectadas.

Realizar efetivamente essas características introduz complexidade significativa. Como o agente mantém estado através de múltiplos passos em sua tela? Como decide quando e como usar uma ferramenta? Como a comunicação entre diferentes agentes é gerenciada? Como você constrói resiliência no sistema para lidar com resultados inesperados ou erros?

## Por que Padrões Importam no Desenvolvimento de Agentes

Esta complexidade é precisamente por que padrões de design agênticos são indispensáveis. Eles não são regras rígidas, mas sim templates ou blueprints testados em batalha que oferecem abordagens comprovadas para desafios padrão de design e implementação no domínio agêntico. Ao reconhecer e aplicar esses design patterns, você ganha acesso a soluções que melhoram a estrutura, manutenibilidade, confiabilidade e eficiência dos agentes que você constrói em sua tela.

Usar design patterns ajuda você a evitar reinventar soluções fundamentais para tarefas como gerenciar fluxo conversacional, integrar capacidades externas ou coordenar múltiplas ações de agentes. Eles fornecem uma linguagem comum e estrutura que torna a lógica do seu agente mais clara e fácil para outros (e você mesmo no futuro) entenderem e manterem. Implementar padrões projetados para tratamento de erros ou gerenciamento de estado contribui diretamente para construir sistemas mais robustos e confiáveis. Aproveitar essas abordagens estabelecidas acelera seu processo de desenvolvimento, permitindo que você foque nos aspectos únicos de sua aplicação em vez da mecânica fundamental do comportamento de agentes.

Este livro extrai 21 padrões de design chave que representam blocos de construção fundamentais e técnicas para construir agentes sofisticados em várias telas técnicas. Compreender e aplicar esses padrões elevará significativamente sua capacidade de projetar e implementar sistemas inteligentes efetivamente.

## Visão Geral do Livro e Como Usá-lo

Este livro, "Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems," é elaborado para ser um recurso prático e acessível. Seu foco principal é explicar claramente cada padrão agêntico e fornecer exemplos de código concretos e executáveis para demonstrar sua implementação. Através de 21 capítulos dedicados, exploraremos uma gama diversa de design patterns, desde conceitos fundamentais como estruturar operações sequenciais (Prompt Chaining) e interação externa (Tool Use) até tópicos mais avançados como trabalho colaborativo (Multi-Agent Collaboration) e auto-aperfeiçoamento (Self-Correction).

O livro é organizado capítulo por capítulo, com cada capítulo mergulhando em um único padrão agêntico. Dentro de cada capítulo, você encontrará:

* Uma Visão Geral do Padrão detalhada fornecendo uma explicação clara do padrão e seu papel no design agêntico.
* Uma seção sobre Aplicações Práticas e Casos de Uso ilustrando cenários do mundo real onde o padrão é inestimável e os benefícios que traz.
* Um Exemplo de Código Hands-On oferecendo código prático e executável que demonstra a implementação do padrão usando frameworks proeminentes de desenvolvimento de agentes. É aqui que você verá como aplicar o padrão dentro do contexto de uma tela técnica.
* Principais Conclusões resumindo os pontos mais cruciais para revisão rápida.
* Referências para exploração adicional, fornecendo recursos para aprendizado mais profundo sobre o padrão e conceitos relacionados.

Embora os capítulos sejam ordenados para construir conceitos progressivamente, sinta-se livre para usar o livro como referência, pulando para capítulos que abordem desafios específicos que você enfrenta em seus próprios projetos de desenvolvimento de agentes. Os apêndices fornecem uma visão abrangente de técnicas avançadas de prompting, princípios para aplicar agentes de IA em ambientes do mundo real, e uma visão geral de frameworks agênticos essenciais. Para complementar isso, tutoriais práticos apenas online são incluídos, oferecendo orientação passo a passo sobre construir agentes com plataformas específicas como AgentSpace e para a interface de linha de comando. A ênfase ao longo é na aplicação prática; encorajamos fortemente que você execute os exemplos de código, experimente com eles e os adapte para construir seus próprios sistemas inteligentes em sua tela escolhida.

Uma ótima pergunta que ouço é: 'Com a IA mudando tão rapidamente, por que escrever um livro que poderia ficar rapidamente desatualizado?' Minha motivação foi na verdade o oposto. É precisamente porque as coisas estão se movendo tão rapidamente que precisamos dar um passo atrás e identificar os princípios subjacentes que estão se solidificando. Padrões como RAG, Reflection, Routing, Memory e outros que discuto, estão se tornando blocos de construção fundamentais. Este livro é um convite para refletir sobre essas ideias centrais, que fornecem a fundação que precisamos para construir. Humanos precisam desses momentos de reflexão sobre padrões fundamentais.

## Introdução aos Frameworks Utilizados

Para fornecer uma "tela" tangível para nossos exemplos de código (veja também Apêndice), utilizaremos principalmente três frameworks proeminentes de desenvolvimento de agentes. LangChain, junto com sua extensão stateful LangGraph, fornece uma maneira flexível de encadear modelos de linguagem e outros componentes, oferecendo uma tela robusta para construir sequências complexas e grafos de operações. Crew AI fornece um framework estruturado especificamente projetado para orquestrar múltiplos agentes de IA, papéis e tarefas, atuando como uma tela particularmente bem adequada para sistemas de agentes colaborativos. O Google Agent Developer Kit (Google ADK) oferece ferramentas e componentes para construir, avaliar e implantar agentes, fornecendo outra tela valiosa, frequentemente integrada com a infraestrutura de IA do Google.

Esses frameworks representam diferentes facetas da tela de desenvolvimento de agentes, cada um com suas forças. Ao mostrar exemplos através dessas ferramentas, você ganhará uma compreensão mais ampla de como os padrões podem ser aplicados independentemente do ambiente técnico específico que você escolher para seus sistemas agênticos. Os exemplos são projetados para ilustrar claramente a lógica central do padrão e sua implementação na tela do framework, focando em clareza e praticidade.

Ao final deste livro, você não apenas compreenderá os conceitos fundamentais por trás de 21 padrões agênticos essenciais, mas também possuirá o conhecimento prático e exemplos de código para aplicá-los efetivamente, permitindo que você construa sistemas mais inteligentes, capazes e autônomos em sua tela de desenvolvimento escolhida. Vamos começar esta jornada hands-on!

---

*De "Agentic Design Patterns" por Antonio Gulli*