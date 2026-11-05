"""Adapter tests for the third-party integrations, using small in-memory stand-ins.

These check that our wrapper code calls ChromaDB, LangChain's
HuggingFaceEmbeddings and LangChain chat models the way their APIs expect.
They run even when those packages are not installed. When the real packages
ARE installed, the fake modules are only injected for the duration of a test.
"""

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import helpers  # noqa: F401

import numpy as np
from pydantic import BaseModel

import rag_index as rag
from config import GUIDELINES_TXT


# ---- Fake chromadb ---------------------------------------------------------

class _FakeCollection:
    def __init__(self, configuration):
        self.configuration = configuration
        self.ids, self.docs, self.embs, self.metas = [], [], [], []

    def add(self, ids, documents, embeddings, metadatas):
        assert len(ids) == len(documents) == len(embeddings) == len(metadatas)
        self.ids += ids
        self.docs += documents
        self.embs += embeddings
        self.metas += metadatas

    def count(self):
        return len(self.ids)

    def query(self, query_embeddings, n_results, include):
        assert set(include) == {"documents", "metadatas", "embeddings"}
        # Rank by squared L2 distance (Chroma's default space) to show ranking does not depend on cosine.
        dists = ((np.array(self.embs) - np.array(query_embeddings[0])) ** 2).sum(axis=1)
        order = np.argsort(dists)[:n_results]
        return {"documents": [[self.docs[i] for i in order]],
                "embeddings": [np.array([self.embs[i] for i in order])],
                "metadatas": [[self.metas[i] for i in order]]}


_STORES = {}


class _FakePersistentClient:
    def __init__(self, path):
        self.collections = _STORES.setdefault(path, {})

    def get_or_create_collection(self, name, configuration=None):
        return self.collections.setdefault(name, _FakeCollection(configuration))

    def delete_collection(self, name):
        if name not in self.collections:
            raise ValueError("does not exist")
        del self.collections[name]


fake_chromadb = types.SimpleNamespace(PersistentClient=_FakePersistentClient)


# ---- Fake langchain_huggingface (a Pydantic model, like the real class) ----

class _FakeHFEmbeddings(BaseModel):
    model_name: str
    encode_kwargs: dict = {}

    def embed_documents(self, texts):
        return rag.HashingEmbeddings().embed_documents(texts)

    def embed_query(self, text):
        return rag.HashingEmbeddings().embed_query(text)


fake_hf = types.SimpleNamespace(HuggingFaceEmbeddings=_FakeHFEmbeddings)


class TestChromaStoreAdapter(unittest.TestCase):
    def test_build_query_and_reload_through_chroma_api(self):
        with mock.patch.dict(sys.modules, {"chromadb": fake_chromadb}), tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                index = rag.build_index(GUIDELINES_TXT, Path(tmp), force=True, embedder=rag.HashingEmbeddings(), log_path=None)
            self.assertEqual(index.store.backend, "chromadb")
            self.assertEqual(index.store.collection.configuration, {"hnsw": {"space": "cosine"}})
            result = index.query("What are the storage requirements below -20C?", 3)
            self.assertEqual(len(result.chunks), 3)
            self.assertEqual(result.similarities, sorted(result.similarities, reverse=True))
            self.assertTrue(all(-1.0 <= sim <= 1.0 for sim in result.similarities))

            reloaded = rag.load_index(Path(tmp), embedder=rag.HashingEmbeddings(), log_path=None)
            self.assertEqual(reloaded.store.count(), index.manifest["n_chunks"])

            with contextlib.redirect_stdout(io.StringIO()):
                rebuilt = rag.build_index(GUIDELINES_TXT, Path(tmp), force=True, embedder=rag.HashingEmbeddings(), log_path=None)
            self.assertEqual(rebuilt.store.count(), index.manifest["n_chunks"])  # reset, not duplicated


class TestHuggingFaceAdapter(unittest.TestCase):
    def test_pydantic_embedding_class_is_wrapped_not_mutated(self):
        with mock.patch.dict(sys.modules, {"langchain_huggingface": fake_hf}), mock.patch.dict("os.environ", {"EMBEDDING_BACKEND": "huggingface"}):
            embedder = rag.get_embedding_model()
        self.assertEqual(embedder.name, rag.DEFAULT_EMBEDDING_MODEL)
        self.assertIsInstance(embedder, rag.LangChainEmbeddings)
        self.assertEqual(len(embedder.embed_documents(["a", "b"])), 2)
        self.assertEqual(embedder.model.encode_kwargs, {"normalize_embeddings": True})

    def test_falls_back_when_package_missing_in_auto_mode(self):
        with mock.patch.dict(sys.modules, {"langchain_huggingface": None}), mock.patch.dict("os.environ", {"EMBEDDING_BACKEND": "auto"}):
            with self.assertWarns(UserWarning):
                embedder = rag.get_embedding_model()
        self.assertIsInstance(embedder, rag.HashingEmbeddings)

    def test_strict_mode_raises_when_package_missing(self):
        with mock.patch.dict(sys.modules, {"langchain_huggingface": None}), mock.patch.dict("os.environ", {"EMBEDDING_BACKEND": "huggingface"}):
            with self.assertRaises(ImportError):
                rag.get_embedding_model()


class TestLangChainChatAdapter(unittest.TestCase):
    def test_anthropic_client_sends_system_and_human_messages(self):
        import llm_providers

        seen = {}

        class FakeChatAnthropic:
            def __init__(self, model, max_tokens, **kwargs):
                seen.update(model=model, max_tokens=max_tokens, extra=kwargs)

            def invoke(self, messages):
                seen["messages"] = messages
                return types.SimpleNamespace(content=[{"type": "text", "text": '{"a": '}, {"type": "text", "text": "1}"}])

        fake_module = types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic)
        with mock.patch.dict(sys.modules, {"langchain_anthropic": fake_module}):
            client = llm_providers.LangChainChatClient("anthropic", model="test-model")
            text = client.complete("SYSTEM", "USER")
        self.assertEqual(text, '{"a": 1}')
        self.assertEqual(seen["messages"], [("system", "SYSTEM"), ("human", "USER")])
        self.assertNotIn("temperature", seen["extra"])  # current Claude models reject sampling parameters
        self.assertGreaterEqual(seen["max_tokens"], 8000)  # room for adaptive thinking before the JSON
        self.assertEqual(client.name, "anthropic:test-model")

    def test_auto_provider_without_keys_is_offline(self):
        import llm_providers

        env = {"LLM_PROVIDER": "auto", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}
        with mock.patch.dict("os.environ", env):
            with contextlib.redirect_stdout(io.StringIO()):
                client = llm_providers.get_llm_client()
        self.assertIsInstance(client, llm_providers.OfflineGroundedClient)


if __name__ == "__main__":
    unittest.main()
