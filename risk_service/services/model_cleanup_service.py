import os
import time
import json
import glob
from typing import Dict, List, Any
from datetime import datetime, timedelta


class ModelCleanupService:
    """
    Serviço para limpeza de modelos antigos e manutenção do registry
    """

    def __init__(self, models_dir: str, registry_path: str):
        self.models_dir = models_dir
        self.registry_path = registry_path

    def cleanup_old_models(self,
                           keep_best_n: int = 10,
                           max_age_days: int = 30,
                           min_accuracy_threshold: float = 0.5,
                           dry_run: bool = False) -> Dict[str, Any]:
        """
        Limpa modelos antigos baseado em critérios

        Args:
            keep_best_n: Manter os N melhores modelos por accuracy
            max_age_days: Remover modelos mais antigos que X dias
            min_accuracy_threshold: Remover modelos com accuracy menor que threshold
            dry_run: Se True, apenas simula sem deletar

        Returns:
            Dict com estatísticas da limpeza
        """

        cleanup_stats = {
            "models_analyzed": 0,
            "models_removed": 0,
            "models_kept": 0,
            "space_freed_mb": 0,
            "removed_files": [],
            "kept_files": [],
            "orphaned_files_removed": 0,
            "registry_cleaned": False
        }

        try:
            # 1. Carrega registry atual
            registry = self._load_registry()
            models_in_registry = registry.get("models", [])
            cleanup_stats["models_analyzed"] = len(models_in_registry)

            if not models_in_registry:
                print("Nenhum modelo encontrado no registry")
                return cleanup_stats

            # 2. Filtra modelos válidos (arquivo existe)
            valid_models = []
            for model in models_in_registry:
                if os.path.exists(model["path"]):
                    # Adiciona informações de arquivo
                    stat = os.stat(model["path"])
                    model["file_size_mb"] = stat.st_size / (1024 * 1024)
                    model["file_age_days"] = (time.time() - stat.st_mtime) / (24 * 3600)
                    valid_models.append(model)
                else:
                    print(f"Modelo órfão no registry (arquivo não existe): {model['path']}")

            # 3. Aplica critérios de limpeza
            models_to_keep = []
            models_to_remove = []

            # Ordena por accuracy (melhor primeiro)
            valid_models.sort(key=lambda x: x["metrics"]["accuracy"], reverse=True)

            for i, model in enumerate(valid_models):
                keep_reasons = []
                remove_reasons = []

                # Critério 1: Top N melhores
                if i < keep_best_n:
                    keep_reasons.append(f"top_{keep_best_n}_best")

                # Critério 2: Idade
                if model["file_age_days"] > max_age_days:
                    remove_reasons.append(f"older_than_{max_age_days}_days")

                # Critério 3: Accuracy mínima
                if model["metrics"]["accuracy"] < min_accuracy_threshold:
                    remove_reasons.append(f"accuracy_below_{min_accuracy_threshold}")

                # Decisão final
                if keep_reasons and not remove_reasons:
                    models_to_keep.append(model)
                elif remove_reasons and not keep_reasons:
                    models_to_remove.append(model)
                elif keep_reasons and remove_reasons:
                    # Conflito: prioriza manter se está no top N
                    if f"top_{keep_best_n}_best" in keep_reasons:
                        models_to_keep.append(model)
                    else:
                        models_to_remove.append(model)
                else:
                    # Sem critérios específicos, mantém
                    models_to_keep.append(model)

            # 4. Remove arquivos (se não for dry_run)
            for model in models_to_remove:
                file_path = model["path"]
                if os.path.exists(file_path):
                    file_size_mb = model["file_size_mb"]

                    if not dry_run:
                        os.remove(file_path)
                        print(f"Removido: {file_path} ({file_size_mb:.2f}MB)")
                    else:
                        print(f"[DRY RUN] Removeria: {file_path} ({file_size_mb:.2f}MB)")

                    cleanup_stats["models_removed"] += 1
                    cleanup_stats["space_freed_mb"] += file_size_mb
                    cleanup_stats["removed_files"].append(file_path)

            # 5. Atualiza registry
            if not dry_run:
                registry["models"] = models_to_keep
                self._save_registry(registry)
                cleanup_stats["registry_cleaned"] = True

            cleanup_stats["models_kept"] = len(models_to_keep)
            cleanup_stats["kept_files"] = [m["path"] for m in models_to_keep]

            # 6. Remove arquivos órfãos (modelos no disco mas não no registry)
            orphaned_count = self._cleanup_orphaned_files(models_to_keep, dry_run)
            cleanup_stats["orphaned_files_removed"] = orphaned_count

            return cleanup_stats

        except Exception as e:
            print(f"Erro durante limpeza: {e}")
            cleanup_stats["error"] = str(e)
            return cleanup_stats

    def _cleanup_orphaned_files(self, valid_models: List[Dict], dry_run: bool) -> int:
        """Remove arquivos .joblib órfãos (não estão no registry)"""

        # Paths dos modelos válidos no registry
        valid_paths = {model["path"] for model in valid_models}

        # Todos os arquivos .joblib no diretório
        all_model_files = glob.glob(os.path.join(self.models_dir, "*.joblib"))

        orphaned_count = 0
        for file_path in all_model_files:
            if file_path not in valid_paths:
                if not dry_run:
                    os.remove(file_path)
                    print(f"Arquivo órfão removido: {file_path}")
                else:
                    print(f"[DRY RUN] Removeria arquivo órfão: {file_path}")
                orphaned_count += 1

        return orphaned_count

    def _load_registry(self) -> Dict[str, Any]:
        """Carrega o registry de modelos"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Erro ao carregar registry: {e}")
                return {"models": [], "best_model": None}
        return {"models": [], "best_model": None}

    def _save_registry(self, registry: Dict[str, Any]):
        """Salva o registry atualizado"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            print(f"Erro ao salvar registry: {e}")

    def get_cleanup_recommendations(self) -> Dict[str, Any]:
        """
        Analisa o estado atual e sugere parâmetros de limpeza
        """

        registry = self._load_registry()
        models = registry.get("models", [])

        if not models:
            return {"recommendation": "no_models_found"}

        # Estatísticas atuais
        total_models = len(models)
        accuracies = [m["metrics"]["accuracy"] for m in models]

        # Calcula idades dos arquivos
        ages_days = []
        total_size_mb = 0

        for model in models:
            if os.path.exists(model["path"]):
                stat = os.stat(model["path"])
                age_days = (time.time() - stat.st_mtime) / (24 * 3600)
                ages_days.append(age_days)
                total_size_mb += stat.st_size / (1024 * 1024)

        recommendations = {
            "total_models": total_models,
            "total_size_mb": round(total_size_mb, 2),
            "accuracy_stats": {
                "min": round(min(accuracies), 3),
                "max": round(max(accuracies), 3),
                "avg": round(sum(accuracies) / len(accuracies), 3)
            },
            "age_stats_days": {
                "min": round(min(ages_days), 1) if ages_days else 0,
                "max": round(max(ages_days), 1) if ages_days else 0,
                "avg": round(sum(ages_days) / len(ages_days), 1) if ages_days else 0
            }
        }

        # Sugestões baseadas no estado atual
        if total_models > 20:
            recommendations["suggested_keep_best_n"] = 10
        elif total_models > 10:
            recommendations["suggested_keep_best_n"] = 5
        else:
            recommendations["suggested_keep_best_n"] = max(3, total_models // 2)

        if max(ages_days) if ages_days else 0 > 60:
            recommendations["suggested_max_age_days"] = 30
        else:
            recommendations["suggested_max_age_days"] = 60

        if min(accuracies) < 0.6:
            recommendations["suggested_min_accuracy"] = 0.6
        else:
            recommendations["suggested_min_accuracy"] = 0.5

        return recommendations

    def cleanup_models_by_pattern(self,
                                  name_pattern: str = None,
                                  older_than_hours: int = 24,
                                  dry_run: bool = True) -> Dict[str, Any]:
        """
        Limpeza específica por padrão de nome ou idade

        Args:
            name_pattern: Padrão no nome do modelo (ex: "model_rf_*")
            older_than_hours: Remove modelos mais antigos que X horas
            dry_run: Simulação
        """

        cleanup_stats = {
            "files_analyzed": 0,
            "files_removed": 0,
            "space_freed_mb": 0,
            "removed_files": []
        }

        try:
            # Busca arquivos por padrão
            if name_pattern:
                pattern_path = os.path.join(self.models_dir, name_pattern)
                files = glob.glob(pattern_path)
            else:
                files = glob.glob(os.path.join(self.models_dir, "*.joblib"))

            cleanup_stats["files_analyzed"] = len(files)

            cutoff_time = time.time() - (older_than_hours * 3600)

            for file_path in files:
                if os.path.exists(file_path):
                    stat = os.stat(file_path)

                    if stat.st_mtime < cutoff_time:
                        file_size_mb = stat.st_size / (1024 * 1024)

                        if not dry_run:
                            os.remove(file_path)
                            print(f"Removido por idade: {file_path}")
                        else:
                            print(f"[DRY RUN] Removeria por idade: {file_path}")

                        cleanup_stats["files_removed"] += 1
                        cleanup_stats["space_freed_mb"] += file_size_mb
                        cleanup_stats["removed_files"].append(file_path)

            return cleanup_stats

        except Exception as e:
            print(f"Erro na limpeza por padrão: {e}")
            cleanup_stats["error"] = str(e)
            return cleanup_stats