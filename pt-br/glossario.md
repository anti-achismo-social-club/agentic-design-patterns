# Glossário

*De "Agentic Design Patterns" por Antonio Gulli*

## Conceitos Fundamentais

**Prompt**: Um prompt é a entrada, tipicamente na forma de uma pergunta, instrução ou declaração, que um usuário fornece a um modelo de IA para provocar uma resposta. A qualidade e estrutura do prompt influenciam pesadamente a saída do modelo, tornando a engenharia de prompts uma habilidade-chave para usar IA efetivamente.

**Context Window (Janela de Contexto)**: A janela de contexto é o número máximo de tokens que um modelo de IA pode processar de uma vez, incluindo tanto a entrada quanto sua saída gerada. Este tamanho fixo é uma limitação crítica, já que informações fora da janela são ignoradas, enquanto janelas maiores permitem conversas mais complexas e análise de documentos.

**In-Context Learning (Aprendizado em Contexto)**: Aprendizado em contexto é a capacidade de uma IA aprender uma nova tarefa a partir de exemplos fornecidos diretamente no prompt, sem exigir qualquer retreinamento. Esta funcionalidade poderosa permite que um modelo único e de propósito geral seja adaptado para incontáveis tarefas específicas dinamicamente.

**Zero-Shot, One-Shot & Few-Shot Prompting**: Estas são técnicas de prompting onde um modelo recebe zero, um, ou alguns exemplos de uma tarefa para guiar sua resposta. Fornecer mais exemplos geralmente ajuda o modelo a entender melhor a intenção do usuário e melhora sua precisão para a tarefa específica.

**Multimodalidade**: Multimodalidade é a capacidade de uma IA entender e processar informações através de múltiplos tipos de dados como texto, imagens e áudio. Isso permite interações mais versáteis e semelhantes às humanas, como descrever uma imagem ou responder uma pergunta falada.

**Grounding (Fundamentação)**: Fundamentação é o processo de conectar as saídas de um modelo a fontes de informação verificáveis do mundo real para garantir precisão factual e reduzir alucinações. Isso é frequentemente alcançado com técnicas como RAG para tornar sistemas de IA mais confiáveis.

## Arquiteturas Principais de Modelos de IA

**Transformers**: O Transformer é a arquitetura de rede neural fundamental para a maioria dos LLMs modernos. Sua inovação-chave é o mecanismo de auto-atenção, que processa eficientemente longas sequências de texto e captura relacionamentos complexos entre palavras.

**Recurrent Neural Network (RNN - Rede Neural Recorrente)**: A Rede Neural Recorrente é uma arquitetura fundamental que precedeu o Transformer. RNNs processam informações sequencialmente, usando loops para manter uma "memória" de entradas anteriores, o que as tornou adequadas para tarefas como processamento de texto e fala.

**Mixture of Experts (MoE - Mistura de Especialistas)**: Mistura de Especialistas é uma arquitetura de modelo eficiente onde uma rede "roteadora" seleciona dinamicamente um pequeno subconjunto de redes "especialistas" para lidar com qualquer entrada dada. Isso permite que modelos tenham um número massivo de parâmetros mantendo custos computacionais gerenciáveis.

**Diffusion Models (Modelos de Difusão)**: Modelos de difusão são modelos generativos que excel em criar imagens de alta qualidade. Eles funcionam adicionando ruído aleatório aos dados e então treinando um modelo para reverter meticulosamente o processo, permitindo gerar dados novos de um ponto de partida aleatório.

## Ciclo de Vida de Desenvolvimento de LLM

O desenvolvimento de um modelo de linguagem poderoso segue uma sequência distinta. Começa com **Pré-treinamento**, onde um modelo base massivo é construído treinando-o em um vasto dataset de texto geral da internet para aprender linguagem, raciocínio e conhecimento mundial. Em seguida vem o **Fine-tuning**, uma fase de especialização onde o modelo geral é treinado ainda mais em datasets menores e específicos de tarefa para adaptar suas capacidades para um propósito particular. O estágio final é **Alinhamento**, onde o comportamento do modelo especializado é ajustado para garantir que suas saídas sejam úteis, inofensivas e alinhadas com valores humanos.

## Aprimorando Capacidades de Agentes de IA

Agentes de IA são sistemas que podem perceber seu ambiente e tomar ações autônomas para alcançar objetivos. Sua eficácia é aprimorada por frameworks robustos de raciocínio.

**Chain of Thought (CoT - Cadeia de Pensamento)**: Esta técnica de prompting encoraja um modelo a explicar seu raciocínio passo a passo antes de dar uma resposta final. Este processo de "pensar em voz alta" frequentemente leva a resultados mais precisos em tarefas complexas de raciocínio.

**Tree of Thoughts (ToT - Árvore de Pensamentos)**: Árvore de Pensamentos é um framework avançado de raciocínio onde um agente explora múltiplos caminhos de raciocínio simultaneamente, como galhos em uma árvore. Permite ao agente auto-avaliar diferentes linhas de pensamento e escolher a mais promissora para prosseguir, tornando-o mais eficaz na resolução de problemas complexos.

**ReAct (Reason and Act - Raciocinar e Agir)**: ReAct é um framework de agente que combina raciocínio e ação em um loop. O agente primeiro "pensa" sobre o que fazer, então toma uma "ação" usando uma ferramenta, e usa a observação resultante para informar seu próximo pensamento, tornando-o altamente eficaz na resolução de tarefas complexas.

**Planning (Planejamento)**: Esta é a capacidade de um agente dividir um objetivo de alto nível em uma sequência de sub-tarefas menores e gerenciáveis. O agente então cria um plano para executar estes passos em ordem, permitindo-lhe lidar com atribuições complexas e multi-passo.

## Termos por Categoria

### A
**Agente Autônomo** - Um sistema que pode operar independentemente para alcançar objetivos específicos sem supervisão humana constante.

**Agente Deliberativo** - Um agente que usa raciocínio explícito e planejamento antes de tomar ações.

**Aprendizado Adaptativo** - A capacidade de modificar comportamento baseado em experiência e feedback.

### B
**Balanceamento de Carga** - Distribuição de tarefas computacionais através de múltiplos recursos para otimizar performance.

**Busca Heurística** - Técnicas de busca que usam regras práticas para encontrar soluções eficientemente.

### C
**Colaboração Multi-Agente** - Coordenação entre múltiplos agentes para alcançar objetivos comuns.

**Comportamento Emergente** - Comportamentos complexos que surgem da interação de comportamentos mais simples de agentes, não explicitamente programados.

**Crítica e Reflexão** - Processo onde agentes avaliam e melhoram suas próprias ações e resultados.

### D
**Degradação Graceful** - Capacidade de um sistema manter funcionalidade básica mesmo quando alguns componentes falham.

**Descoberta de Conhecimento** - Processo de extrair informações úteis e padrões de dados.

### E
**Escalabilidade** - Capacidade de um sistema manter performance conforme cresce em tamanho ou complexidade.

**Exploração vs Exploração** - O trade-off entre tentar novas ações (exploração) e aproveitar ações conhecidas bem-sucedidas (exploração).

### F
**Fail-Safe** - Mecanismos que garantem operação segura mesmo durante falhas.

**Feedback Loop** - Processo onde saídas de um sistema são alimentadas de volta como entradas para influenciar comportamento futuro.

**Function Calling** - A capacidade de um agente invocar ferramentas externas, APIs ou serviços para estender suas capacidades.

### G
**Gerenciamento de Estado** - Manutenção e atualizações de informações sobre o estado atual do sistema.

**Guardrails** - Mecanismos de segurança que restringem comportamento do agente para prevenir ações prejudiciais ou não intencionais.

### H
**Human-in-the-Loop (HITL)** - Padrões de design que incorporam supervisão e intervenção humana em processos de tomada de decisão de agentes.

**Híbrido Simbólico-Neural** - Abordagens que combinam métodos de IA simbólicos e neurais.

### I
**Inteligência Distribuída** - Inteligência que emerge da coordenação de múltiplos agentes ou componentes.

**Interface Conversacional** - Sistemas que permitem interação através de linguagem natural.

### M
**Memória Episódica** - Um tipo de memória que armazena experiências e eventos específicos com informações temporais e contextuais.

**Monitoramento em Tempo Real** - Observação contínua de sistemas para detectar problemas e otimizar performance.

### O
**Orquestração** - Coordenação de múltiplos serviços ou agentes para alcançar um objetivo comum.

**Otimização Multi-Objetivo** - Balanceamento de múltiplos objetivos potencialmente conflitantes.

### P
**Padrão de Design** - Soluções reutilizáveis para problemas comuns no desenvolvimento de software.

**Planejamento Hierárquico** - Uma abordagem de planejamento que divide objetivos complexos em múltiplos níveis de sub-objetivos e sub-tarefas.

**Priorização Dinâmica** - A capacidade de ajustar prioridades de tarefas em tempo real baseado em condições e requisitos em mudança.

### Q
**Qualidade de Serviço (QoS)** - Medidas de performance e confiabilidade de um sistema.

### R
**Raciocínio Causal** - Compreensão de relacionamentos causa-efeito entre eventos e ações.

**Recuperação de Erros** - Mecanismos que permitem a agentes detectar, lidar e recuperar de vários tipos de falhas.

**Resiliência** - Capacidade de um sistema se recuperar de falhas e continuar operando.

### S
**Sistema Adaptativo** - Sistema que pode modificar seu comportamento baseado em mudanças ambientais.

**Sistema Reativo** - Sistema que responde a estímulos externos ou mudanças ambientais.

### T
**Tool Use (Uso de Ferramentas)** - Capacidade de agentes utilizarem recursos externos para estender suas capacidades.

**Tomada de Decisão Distribuída** - Processo onde decisões são feitas através de múltiplos agentes ou componentes.

### V
**Validação** - Processo de verificar que um sistema atende aos requisitos especificados.

---

*Este glossário fornece definições essenciais para entender padrões de design agentic e conceitos relacionados.*

---

**Nota de Tradução**: Este documento foi traduzido do inglês para português brasileiro. Termos técnicos foram mantidos em inglês quando amplamente estabelecidos na comunidade técnica brasileira.