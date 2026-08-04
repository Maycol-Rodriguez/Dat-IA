# Despliegue de Dat-IA en Kubernetes

Guía breve para desplegar y comprobar Dat-IA 0.2.0 en un clúster local `kind`
sobre Docker Desktop.

## Recursos desplegados

| Recurso | Propósito |
|---|---|
| Namespace `dat-ia` | Aislamiento |
| Deployment `dat-ia-api` | Administración y autorreparación del Pod |
| Service `dat-ia-api` | Acceso interno por `ClusterIP` |
| ConfigMap `dat-ia-config` | Configuración no sensible |
| Secret `dat-ia-secrets` | Credenciales creadas localmente |
| PVC `dat-ia-chroma` | Persistencia de ChromaDB |
| Probes | Inicio, disponibilidad y salud |

La imagen utilizada es:

```text
ghcr.io/maycol-rodriguez/dat-ia:0.2.0
```

## Requisitos

- Docker Desktop en modo Linux containers.
- `kubectl`.
- `kind`.
- `.env` configurado en la raíz del repositorio.

El archivo `.env` no se versiona.

## 1. Crear el clúster y cargar la imagen

```powershell
kind create cluster `
    --name dat-ia `
    --wait 5m

$image = "ghcr.io/maycol-rodriguez/dat-ia:0.2.0"

docker pull $image

kind load docker-image `
    $image `
    --name dat-ia
```

Verificar:

```powershell
kubectl get nodes
```

El nodo debe aparecer como `Ready`.

## 2. Crear el namespace y el Secret

```powershell
kubectl apply `
    -f k8s/base/namespace.yaml

kubectl create secret generic dat-ia-secrets `
    --namespace dat-ia `
    --from-env-file=".env" `
    --dry-run=client `
    -o yaml |
kubectl apply -f -
```

Los valores del Secret no se guardan en Git.

## 3. Validar y desplegar

```powershell
kubectl apply `
    --dry-run=server `
    -k k8s/base

kubectl apply `
    -k k8s/base

kubectl rollout status `
    deployment/dat-ia-api `
    --namespace dat-ia `
    --timeout=10m
```

Comprobar los recursos:

```powershell
kubectl get `
    deployment,pod,service,pvc `
    --namespace dat-ia `
    -o wide
```

Resultado requerido:

```text
Deployment   1/1 disponible
Pod          1/1 Running, 0 reinicios
Service      ClusterIP
PVC          Bound
```

## 4. Comprobar seguridad y versión

```powershell
$podName = kubectl get pods `
    --namespace dat-ia `
    -l app.kubernetes.io/name=dat-ia-api `
    -o jsonpath='{.items[0].metadata.name}'

kubectl exec `
    $podName `
    --namespace dat-ia `
    -- id

kubectl get pod `
    $podName `
    --namespace dat-ia `
    -o jsonpath='{.spec.containers[0].image}'

Write-Host
```

Resultados esperados:

```text
uid=10001(app) gid=10001(app)
ghcr.io/maycol-rodriguez/dat-ia:0.2.0
```

## 5. Comprobar la API

En una segunda terminal:

```powershell
kubectl port-forward `
    service/dat-ia-api `
    8000:80 `
    --namespace dat-ia
```

En la terminal principal:

```powershell
$health = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/health"

$ready = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/ready"

$health | Format-List
$ready | Format-List
```

Resultados requeridos:

```text
/health -> status: ok, version: 0.2.0
/ready  -> status: ok, database: connected
```

## 6. Comprobar el flujo Text-to-SQL

```powershell
$body = @{
    question = "Cuantas ordenes se registraron en total?"
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/query/answer" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

[PSCustomObject]@{
    Status = $result.status
    SQL    = $result.sql
    Count  = $result.data.order_count
} | Format-List
```

Resultado de referencia:

```text
Status : success
SQL    : SELECT COUNT(order_id) AS order_count FROM olist_orders_dataset
Count  : 99441
```

## 7. Comprobar autorreparación y persistencia

Esta prueba elimina el Pod, no el Deployment ni el PVC.

```powershell
$oldPod = kubectl get pods `
    --namespace dat-ia `
    -l app.kubernetes.io/name=dat-ia-api `
    -o jsonpath='{.items[0].metadata.name}'

kubectl delete pod `
    $oldPod `
    --namespace dat-ia

kubectl wait `
    --for=condition=Ready `
    pod `
    --selector app.kubernetes.io/name=dat-ia-api `
    --namespace dat-ia `
    --timeout=10m

$newPod = kubectl get pods `
    --namespace dat-ia `
    -l app.kubernetes.io/name=dat-ia-api `
    -o jsonpath='{.items[0].metadata.name}'

[PSCustomObject]@{
    PreviousPod = $oldPod
    CurrentPod  = $newPod
    Recreated   = $oldPod -ne $newPod
} | Format-List
```

`Recreated` debe ser `True`.

Verificar la persistencia:

```powershell
kubectl get pvc `
    dat-ia-chroma `
    --namespace dat-ia

kubectl logs `
    $newPod `
    --namespace dat-ia `
    --tail=100 |
Select-String `
    -Pattern `
    'ChromaDB:|Query memory V2:|SQLDatabase|Application startup'
```

El PVC debe continuar `Bound`. El nuevo Pod debe recuperar los 16 esquemas de
ChromaDB y al menos una consulta de Query Memory V2.

## 8. Eliminar el entorno

```powershell
kind delete cluster `
    --name dat-ia
```

## Alcance

Este despliegue demuestra orquestación local, autorreparación, persistencia,
probes, límites de recursos y ejecución sin privilegios. No expone Dat-IA
públicamente en Internet; el acceso local se realiza con `kubectl port-forward`.