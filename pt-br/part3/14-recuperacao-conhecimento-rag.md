# Capítulo 14: Recuperação de Conhecimento (RAG)

Geração Aumentada por Recuperação (RAG) é um padrão de design poderoso que aprimora agentes de IA fornecendo acesso a fontes de conhecimento externas, permitindo que gerem respostas mais precisas, atualizadas e contextualmente relevantes recuperando e incorporando informações relevantes durante o processo de geração.

## Introdução

A Geração Aumentada por Recuperação (RAG) representa uma mudança de paradigma em como agentes de IA acessam e utilizam conhecimento. Modelos de linguagem tradicionais são limitados pelo corte temporal de seus dados de treinamento e não conseguem acessar informações em tempo real ou conhecimento específico de domínio que não estava presente durante o treinamento. RAG aborda essas limitações combinando as capacidades generativas de grandes modelos de linguagem com recuperação dinâmica de informações de fontes de conhecimento externas.

O padrão RAG permite que agentes de IA consultem documentos relevantes, bancos de dados, APIs ou outros repositórios de conhecimento durante o processo de geração, expandindo significativamente sua base de conhecimento e melhorando a precisão das respostas. Esta abordagem é particularmente valiosa para aplicações que requerem informações atuais, conhecimento especializado de domínio ou acesso a dados organizacionais privados.

Sistemas RAG operam em um princípio simples, mas poderoso: quando confrontados com uma consulta, o sistema primeiro recupera informações relevantes de fontes de conhecimento, depois usa este contexto recuperado para gerar respostas mais informadas e precisas. Este processo de duas etapas - recuperação seguida por geração aumentada - cria agentes de IA que podem fornecer respostas factuais, atuais e contextualmente apropriadas mantendo a fluência e capacidades de raciocínio de grandes modelos de linguagem.

O padrão tornou-se essencial para aplicações empresariais de IA, sistemas de perguntas e respostas, chatbots com acesso a bases de conhecimento da empresa e qualquer cenário onde agentes de IA precisam trabalhar com informações em evolução ou especializadas que se estendem além de seus dados de treinamento.

## Conceitos-Chave

### Mecanismos de Recuperação
Diferentes abordagens para encontrar informações relevantes:

- **Recuperação Densa**: Usar modelos de embedding para encontrar conteúdo semanticamente similar
- **Recuperação Esparsa**: Busca tradicional baseada em palavras-chave usando técnicas como BM25
- **Recuperação Híbrida**: Combinar métodos densos e esparsos para melhor cobertura
- **Recuperação Hierárquica**: Recuperação multi-estágio para grandes bases de conhecimento
- **Recuperação Baseada em Grafos**: Aproveitar grafos de conhecimento e relacionamentos de entidades

### Fontes de Conhecimento
Vários tipos de repositórios de conhecimento externos:

- **Coleções de Documentos**: PDFs, páginas web, artigos de pesquisa, manuais
- **Bancos de Dados Estruturados**: Bancos SQL, grafos de conhecimento, APIs
- **Dados em Tempo Real**: Feeds de notícias, dados de sensores, informações de mercado
- **Sistemas Empresariais**: CRM, ERP, documentação interna
- **Busca Web**: Resultados dinâmicos de busca web e conteúdo rastreado

### Gerenciamento de Contexto
Lidar efetivamente com informações recuperadas:

- **Classificação de Contexto**: Ordenar informações recuperadas por relevância
- **Fusão de Contexto**: Combinar múltiplas fontes de informação
- **Filtragem de Contexto**: Remover informações irrelevantes ou contraditórias
- **Compressão de Contexto**: Resumir grandes quantidades de conteúdo recuperado
- **Validação de Contexto**: Verificar a precisão e confiabilidade das fontes

### Aprimoramento de Geração
Melhorar respostas geradas com contexto recuperado:

- **Engenharia de Prompt**: Estruturar prompts para usar efetivamente contexto recuperado
- **Integração de Citações**: Incorporar referências de fontes em respostas geradas
- **Pontuação de Confiança**: Avaliar a confiabilidade de respostas geradas
- **Verificação de Fatos**: Verificar cruzado de conteúdo gerado contra fontes

## Implementação

### Arquitetura RAG Básica

```python
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank: int

class RAGSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.knowledge_base = []
        self.document_index = {}

    async def add_documents(self, documents: List[Document]):
        """Adicionar documentos à base de conhecimento com embeddings"""
        for doc in documents:
            # Gerar embedding para o documento
            doc.embedding = self.embedding_model.encode(doc.content)
            self.knowledge_base.append(doc)
            self.document_index[doc.id] = doc

    async def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Recuperar documentos relevantes para uma consulta"""
        # Gerar embedding da consulta
        query_embedding = self.embedding_model.encode(query)

        # Calcular similaridades
        similarities = []
        for doc in self.knowledge_base:
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            similarities.append((doc, similarity))

        # Ordenar por similaridade e retornar top_k
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc, score) in enumerate(similarities[:top_k]):
            results.append(RetrievalResult(
                document=doc,
                score=score,
                rank=rank
            ))

        return results

    async def generate_with_retrieval(self, query: str, llm_client) -> Dict[str, Any]:
        """Gerar resposta usando contexto recuperado"""
        # Recuperar documentos relevantes
        retrieved_docs = await self.retrieve(query)

        # Preparar contexto a partir de documentos recuperados
        context = self._prepare_context(retrieved_docs)

        # Gerar resposta com contexto
        response = await self._generate_response(query, context, llm_client)

        return {
            "query": query,
            "response": response,
            "sources": [
                {
                    "id": result.document.id,
                    "score": result.score,
                    "content": result.document.content[:200] + "...",
                    "metadata": result.document.metadata
                }
                for result in retrieved_docs
            ]
        }

    def _prepare_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Preparar string de contexto a partir de documentos recuperados"""
        context_parts = []

        for result in retrieved_docs:
            doc = result.document
            context_part = f"Fonte {result.rank + 1} (Pontuação: {result.score:.3f}):\n"
            context_part += f"Título: {doc.metadata.get('title', 'N/A')}\n"
            context_part += f"Conteúdo: {doc.content}\n"
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    async def _generate_response(self, query: str, context: str, llm_client) -> str:
        """Gerar resposta usando LLM com contexto recuperado"""
        prompt = f"""
        Use o seguinte contexto para responder à pergunta. Se a resposta não puder ser encontrada no contexto, diga isso claramente.

        Contexto:
        {context}

        Pergunta: {query}

        Resposta:
        """

        response = await llm_client.complete(prompt)
        return response
```

### Sistema de Recuperação Híbrida Avançado

```python
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridRetriever:
    def __init__(self, dense_weight=0.7, sparse_weight=0.3):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.documents = []
        self.tfidf_matrix = None

    async def index_documents(self, documents: List[Document]):
        """Indexar documentos para recuperação densa e esparsa"""
        self.documents = documents

        # Preparar embeddings para recuperação densa
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(contents)

        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        # Preparar matriz TF-IDF para recuperação esparsa
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)

    async def hybrid_retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Realizar recuperação híbrida combinando métodos densos e esparsos"""
        # Pontuações de recuperação densa
        dense_scores = await self._dense_retrieve(query)

        # Pontuações de recuperação esparsa
        sparse_scores = await self._sparse_retrieve(query)

        # Combinar pontuações
        combined_scores = self._combine_scores(dense_scores, sparse_scores)

        # Ordenar e retornar resultados top_k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_idx, score) in enumerate(sorted_results[:top_k]):
            results.append(RetrievalResult(
                document=self.documents[doc_idx],
                score=score,
                rank=rank
            ))

        return results

    async def _dense_retrieve(self, query: str) -> Dict[int, float]:
        """Recuperação densa usando embeddings"""
        query_embedding = self.embedding_model.encode(query)
        scores = {}

        for idx, doc in enumerate(self.documents):
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            scores[idx] = similarity

        return scores

    async def _sparse_retrieve(self, query: str) -> Dict[int, float]:
        """Recuperação esparsa usando TF-IDF"""
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = (self.tfidf_matrix * query_vector.T).toarray().flatten()

        scores = {}
        for idx, similarity in enumerate(similarities):
            scores[idx] = similarity

        return scores

    def _combine_scores(self, dense_scores: Dict[int, float], sparse_scores: Dict[int, float]) -> Dict[int, float]:
        """Combinar pontuações densas e esparsas com ponderação"""
        # Normalizar pontuações para faixa [0, 1]
        dense_scores = self._normalize_scores(dense_scores)
        sparse_scores = self._normalize_scores(sparse_scores)

        combined_scores = {}
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())

        for idx in all_indices:
            dense_score = dense_scores.get(idx, 0)
            sparse_score = sparse_scores.get(idx, 0)

            combined_score = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score
            )
            combined_scores[idx] = combined_score

        return combined_scores

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalizar pontuações para faixa [0, 1]"""
        if not scores:
            return scores

        min_score = min(scores.values())
        max_score = max(scores.values())

        if max_score == min_score:
            return {k: 1.0 for k in scores.keys()}

        normalized = {}
        for idx, score in scores.items():
            normalized[idx] = (score - min_score) / (max_score - min_score)

        return normalized
```

### RAG Hierárquico para Grandes Bases de Conhecimento

```python
class HierarchicalRAG:
    def __init__(self):
        self.cluster_embeddings = {}
        self.cluster_documents = {}
        self.document_clusters = {}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def build_hierarchy(self, documents: List[Document], num_clusters=10):
        """Construir estrutura hierárquica para recuperação eficiente"""
        from sklearn.cluster import KMeans

        # Gerar embeddings para todos os documentos
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(contents)

        # Agrupar documentos
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Organizar documentos por cluster
        for doc, embedding, cluster_id in zip(documents, embeddings, cluster_labels):
            doc.embedding = embedding

            if cluster_id not in self.cluster_documents:
                self.cluster_documents[cluster_id] = []

            self.cluster_documents[cluster_id].append(doc)
            self.document_clusters[doc.id] = cluster_id

        # Criar embeddings de cluster (centroides)
        for cluster_id in self.cluster_documents:
            cluster_docs = self.cluster_documents[cluster_id]
            cluster_embeddings = [doc.embedding for doc in cluster_docs]
            centroid = np.mean(cluster_embeddings, axis=0)
            self.cluster_embeddings[cluster_id] = centroid

    async def hierarchical_retrieve(self, query: str, top_clusters=3, docs_per_cluster=5) -> List[RetrievalResult]:
        """Realizar recuperação hierárquica: primeiro clusters, depois documentos"""
        # Estágio 1: Recuperar clusters relevantes
        query_embedding = self.embedding_model.encode(query)
        cluster_scores = []

        for cluster_id, cluster_embedding in self.cluster_embeddings.items():
            similarity = np.dot(query_embedding, cluster_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cluster_embedding)
            )
            cluster_scores.append((cluster_id, similarity))

        # Ordenar clusters por relevância
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        top_cluster_ids = [cluster_id for cluster_id, _ in cluster_scores[:top_clusters]]

        # Estágio 2: Recuperar documentos dos top clusters
        all_results = []
        rank = 0

        for cluster_id in top_cluster_ids:
            cluster_docs = self.cluster_documents[cluster_id]

            # Calcular pontuações de documentos dentro do cluster
            doc_scores = []
            for doc in cluster_docs:
                similarity = np.dot(query_embedding, doc.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
                )
                doc_scores.append((doc, similarity))

            # Ordenar documentos por relevância dentro do cluster
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Adicionar top documentos deste cluster
            for doc, score in doc_scores[:docs_per_cluster]:
                all_results.append(RetrievalResult(
                    document=doc,
                    score=score,
                    rank=rank
                ))
                rank += 1

        # Ordenação final por pontuação entre todos os clusters
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Atualizar ranks
        for rank, result in enumerate(all_results):
            result.rank = rank

        return all_results
```

## Exemplos de Código

### Sistema RAG Completo com Múltiplas Fontes de Conhecimento

```python
import aiohttp
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json

class KnowledgeSource(ABC):
    """Classe base abstrata para fontes de conhecimento"""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[Document]:
        pass

class DocumentStoreSource(KnowledgeSource):
    """Fonte de conhecimento de armazenamento de documentos"""

    def __init__(self, documents: List[Document]):
        self.rag_system = RAGSystem()
        asyncio.create_task(self.rag_system.add_documents(documents))

    async def search(self, query: str, max_results: int = 5) -> List[Document]:
        results = await self.rag_system.retrieve(query, top_k=max_results)
        return [result.document for result in results]

class WebSearchSource(KnowledgeSource):
    """Fonte de conhecimento de busca web"""

    def __init__(self, search_api_key: str):
        self.api_key = search_api_key

    async def search(self, query: str, max_results: int = 5) -> List[Document]:
        # Implementar chamada de API de busca web
        async with aiohttp.ClientSession() as session:
            search_url = f"https://api.search.com/search?q={query}&key={self.api_key}"
            async with session.get(search_url) as response:
                search_results = await response.json()

        documents = []
        for i, result in enumerate(search_results.get("results", [])[:max_results]):
            doc = Document(
                id=f"web_{i}",
                content=result["snippet"],
                metadata={
                    "title": result["title"],
                    "url": result["url"],
                    "source": "web_search"
                }
            )
            documents.append(doc)

        return documents

class DatabaseSource(KnowledgeSource):
    """Fonte de conhecimento de banco de dados"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def search(self, query: str, max_results: int = 5) -> List[Document]:
        # Implementar busca em banco de dados
        # Este é um exemplo simplificado
        sql_query = f"""
        SELECT id, content, title, category
        FROM knowledge_articles
        WHERE content LIKE '%{query}%' OR title LIKE '%{query}%'
        LIMIT {max_results}
        """

        # Executar consulta (pseudo-código)
        results = await self._execute_query(sql_query)

        documents = []
        for result in results:
            doc = Document(
                id=f"db_{result['id']}",
                content=result["content"],
                metadata={
                    "title": result["title"],
                    "category": result["category"],
                    "source": "database"
                }
            )
            documents.append(doc)

        return documents

    async def _execute_query(self, query: str):
        # Implementar execução real de banco de dados
        pass

class MultiSourceRAG:
    """Sistema RAG que combina múltiplas fontes de conhecimento"""

    def __init__(self):
        self.knowledge_sources: List[KnowledgeSource] = []
        self.source_weights = {}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_knowledge_source(self, source: KnowledgeSource, weight: float = 1.0):
        """Adicionar fonte de conhecimento com ponderação opcional"""
        self.knowledge_sources.append(source)
        self.source_weights[len(self.knowledge_sources) - 1] = weight

    async def multi_source_retrieve(self, query: str, max_per_source: int = 3) -> List[Document]:
        """Recuperar documentos de todas as fontes de conhecimento"""
        all_documents = []

        # Recuperar de todas as fontes em paralelo
        search_tasks = []
        for source in self.knowledge_sources:
            task = source.search(query, max_per_source)
            search_tasks.append(task)

        source_results = await asyncio.gather(*search_tasks)

        # Combinar e ponderar resultados
        for source_idx, documents in enumerate(source_results):
            weight = self.source_weights.get(source_idx, 1.0)

            for doc in documents:
                # Adicionar ponderação de fonte aos metadados
                doc.metadata["source_weight"] = weight
                doc.metadata["source_index"] = source_idx
                all_documents.append(doc)

        # Re-classificar documentos entre todas as fontes
        return await self._cross_source_ranking(query, all_documents)

    async def _cross_source_ranking(self, query: str, documents: List[Document]) -> List[Document]:
        """Re-classificar documentos de múltiplas fontes"""
        if not documents:
            return documents

        # Gerar embedding da consulta
        query_embedding = self.embedding_model.encode(query)

        # Calcular pontuações para todos os documentos
        doc_scores = []
        for doc in documents:
            # Gerar embedding do documento se não estiver presente
            if doc.embedding is None:
                doc.embedding = self.embedding_model.encode(doc.content)

            # Calcular similaridade
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )

            # Aplicar ponderação de fonte
            weighted_score = similarity * doc.metadata.get("source_weight", 1.0)
            doc_scores.append((doc, weighted_score))

        # Ordenar por pontuação ponderada
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in doc_scores]

    async def generate_comprehensive_response(self, query: str, llm_client, max_documents: int = 10) -> Dict[str, Any]:
        """Gerar resposta usando múltiplas fontes de conhecimento"""
        # Recuperar documentos de todas as fontes
        retrieved_docs = await self.multi_source_retrieve(query)
        retrieved_docs = retrieved_docs[:max_documents]

        # Agrupar documentos por fonte para organização de contexto
        source_groups = {}
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)

        # Preparar contexto estruturado
        context = self._prepare_multi_source_context(source_groups, query)

        # Gerar resposta
        response = await self._generate_comprehensive_response(query, context, llm_client)

        return {
            "query": query,
            "response": response,
            "sources_used": list(source_groups.keys()),
            "total_documents": len(retrieved_docs),
            "source_breakdown": {
                source: len(docs) for source, docs in source_groups.items()
            },
            "documents": [
                {
                    "id": doc.id,
                    "content_preview": doc.content[:150] + "...",
                    "source": doc.metadata.get("source", "unknown"),
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ]
        }

    def _prepare_multi_source_context(self, source_groups: Dict[str, List[Document]], query: str) -> str:
        """Preparar contexto organizado por fonte"""
        context_parts = [f"Consulta: {query}\n"]

        for source, documents in source_groups.items():
            context_parts.append(f"\n=== Informações de {source.upper()} ===")

            for i, doc in enumerate(documents, 1):
                title = doc.metadata.get("title", "Sem título")
                context_parts.append(f"\n{source.upper()} Fonte {i}: {title}")
                context_parts.append(f"Conteúdo: {doc.content}")

                if "url" in doc.metadata:
                    context_parts.append(f"URL: {doc.metadata['url']}")

        return "\n".join(context_parts)

    async def _generate_comprehensive_response(self, query: str, context: str, llm_client) -> str:
        """Gerar resposta abrangente com atribuição de fonte"""
        prompt = f"""
        Baseado nas seguintes informações de múltiplas fontes, forneça uma resposta abrangente à consulta.
        Inclua detalhes relevantes de diferentes fontes e indique quais fontes apoiam diferentes pontos.
        Se as fontes se contradizerem, mencione a discrepância.
        Sempre cite suas fontes claramente.

        {context}

        Por favor, forneça uma resposta detalhada e bem estruturada que sintetize informações das fontes disponíveis.
        """

        response = await llm_client.complete(prompt)
        return response
```

### RAG com Atualizações de Contexto em Tempo Real

```python
import time
from datetime import datetime, timedelta

class RealTimeRAG:
    """Sistema RAG com atualizações de contexto em tempo real e gerenciamento de cache"""

    def __init__(self, cache_ttl_seconds=3600):
        self.cache_ttl = cache_ttl_seconds
        self.document_cache = {}
        self.query_cache = {}
        self.knowledge_sources = []

    async def add_real_time_source(self, source: KnowledgeSource, update_frequency: int):
        """Adicionar fonte de conhecimento com frequência de atualização especificada (segundos)"""
        self.knowledge_sources.append({
            "source": source,
            "update_frequency": update_frequency,
            "last_update": 0
        })

    async def retrieve_with_freshness(self, query: str, max_staleness_seconds: int = 3600) -> List[Document]:
        """Recuperar documentos garantindo requisitos de frescor"""
        current_time = time.time()

        # Verificar se temos resultados frescos em cache
        if query in self.query_cache:
            cache_entry = self.query_cache[query]
            if current_time - cache_entry["timestamp"] < max_staleness_seconds:
                return cache_entry["documents"]

        # Atualizar fontes obsoletas
        await self._update_stale_sources(current_time)

        # Realizar recuperação fresca
        fresh_documents = await self._fresh_retrieval(query)

        # Cachear resultados
        self.query_cache[query] = {
            "documents": fresh_documents,
            "timestamp": current_time
        }

        return fresh_documents

    async def _update_stale_sources(self, current_time: float):
        """Atualizar fontes de conhecimento que estão obsoletas"""
        update_tasks = []

        for source_info in self.knowledge_sources:
            if current_time - source_info["last_update"] > source_info["update_frequency"]:
                task = self._update_source(source_info, current_time)
                update_tasks.append(task)

        if update_tasks:
            await asyncio.gather(*update_tasks)

    async def _update_source(self, source_info: Dict, current_time: float):
        """Atualizar uma fonte de conhecimento específica"""
        try:
            # Obter documentos mais recentes da fonte
            latest_docs = await source_info["source"].get_latest_documents()

            # Atualizar cache
            for doc in latest_docs:
                self.document_cache[doc.id] = {
                    "document": doc,
                    "timestamp": current_time,
                    "source": source_info["source"]
                }

            source_info["last_update"] = current_time

        except Exception as e:
            print(f"Erro ao atualizar fonte: {e}")

    async def _fresh_retrieval(self, query: str) -> List[Document]:
        """Realizar recuperação fresca em todas as fontes"""
        all_documents = []

        for source_info in self.knowledge_sources:
            try:
                docs = await source_info["source"].search(query)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Erro ao recuperar da fonte: {e}")

        return all_documents

    def cleanup_cache(self):
        """Remover entradas de cache expiradas"""
        current_time = time.time()

        # Limpar cache de documentos
        expired_docs = [
            doc_id for doc_id, cache_entry in self.document_cache.items()
            if current_time - cache_entry["timestamp"] > self.cache_ttl
        ]

        for doc_id in expired_docs:
            del self.document_cache[doc_id]

        # Limpar cache de consultas
        expired_queries = [
            query for query, cache_entry in self.query_cache.items()
            if current_time - cache_entry["timestamp"] > self.cache_ttl
        ]

        for query in expired_queries:
            del self.query_cache[query]
```

## Melhores Práticas

### Otimização de Qualidade de Recuperação
- **Gerenciamento de Tamanho de Chunk**: Otimizar estratégias de chunking de documentos para equilibrar completude de contexto com precisão de recuperação
- **Seleção de Modelo de Embedding**: Escolher modelos de embedding apropriados para seu domínio e requisitos de linguagem
- **Aprimoramento de Consulta**: Implementar técnicas de expansão e reformulação de consulta para melhorar cobertura de recuperação
- **Ajuste de Relevância**: Ajustar continuamente parâmetros de recuperação baseados em métricas de avaliação e feedback do usuário

### Gerenciamento de Contexto
- **Limites de Comprimento de Contexto**: Gerenciar janelas de contexto efetivamente para evitar exceder limitações do modelo maximizando informações relevantes
- **Priorização de Fontes**: Implementar sistemas de ponderação para priorizar fontes de alta qualidade ou autoritativas
- **Tratamento de Redundância**: Detectar e gerenciar conteúdo duplicado ou altamente similar recuperado
- **Compressão de Contexto**: Usar técnicas de sumarização para grandes conjuntos de conteúdo recuperado

### Otimização de Desempenho
- **Estratégias de Cache**: Implementar cache inteligente para documentos e consultas acessados frequentemente
- **Processamento Paralelo**: Utilizar processamento assíncrono para múltiplas fontes de recuperação
- **Otimização de Índice**: Manter índices vetoriais e estruturas de busca eficientes
- **Balanceamento de Carga**: Distribuir requisições de recuperação entre múltiplos sistemas ou réplicas

### Garantia de Qualidade
- **Validação de Fontes**: Verificar a confiabilidade e precisão das fontes de conhecimento
- **Verificação de Respostas**: Implementar mecanismos de verificação de fatos e validação de consistência
- **Precisão de Citações**: Garantir atribuição adequada e rastreabilidade aos materiais fonte
- **Métricas de Avaliação**: Estabelecer métricas abrangentes para qualidade de recuperação e geração

## Armadilhas Comuns

### Pré-processamento Inadequado de Documentos
Chunking, limpeza ou estruturação inadequados de documentos podem impactar significativamente a qualidade da recuperação. Invista tempo em pipelines adequados de pré-processamento de documentos que preservem contexto criando unidades recuperáveis.

### Incompatibilidade Recuperação-Geração
Usar modelos de embedding diferentes para recuperação e geração, ou falhar em alinhar a estratégia de recuperação com as capacidades do modelo de geração, pode levar a desempenho subótimo.

### Sobrecarga de Contexto
Incluir muito contexto recuperado pode sobrecarregar o modelo de geração e levar a respostas desfocadas ou alucinadas. Equilibre riqueza de contexto com capacidade do modelo.

### Negligência de Qualidade de Fontes
Falhar em manter fontes de conhecimento de alta qualidade ou não implementar validação de fontes pode propagar informações imprecisas através do sistema.

### Informações Obsoletas
Não implementar invalidação adequada de cache ou gerenciamento de frescor de conteúdo pode resultar em informações desatualizadas sendo recuperadas e usadas em respostas.

### Pontos Cegos de Avaliação
Focar apenas em métricas de recuperação sem avaliar qualidade de geração end-to-end pode perder problemas importantes no pipeline RAG.

### Negligência de Escalabilidade
Não planejar para escala em termos de volume de documentos, carga de consultas ou latência de recuperação pode levar a gargalos de desempenho conforme o sistema cresce.

---

*Este capítulo cobre 17 páginas de conteúdo de "Agentic Design Patterns" por Antonio Gulli, fornecendo cobertura abrangente de padrões de Geração Aumentada por Recuperação para construir sistemas de agentes de IA aprimorados por conhecimento.*

---

*Nota de Tradução: Este capítulo foi traduzido do inglês para o português brasileiro. Alguns termos técnicos podem ter múltiplas traduções aceitas na literatura em português.*