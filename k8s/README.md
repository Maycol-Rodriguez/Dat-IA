# Despliegue de Dat-IA en Kubernetes

Este directorio contiene los manifiestos necesarios para desplegar la API
Dat-IA 0.2.0 en Kubernetes.

El despliegue fue validado localmente utilizando kind sobre Docker Desktop.

## Arquitectura

El despliegue utiliza los siguientes recursos:

- Namespace `dat-ia`.
- Deployment `dat-ia-api` con una réplica.
- Service `ClusterIP` para exponer internamente la API.
- ConfigMap para configuración no sensible.
- Secret creado localmente desde `.env`.
- PersistentVolumeClaim de 2 GiB para ChromaDB.
- Volumen temporal para la caché de Hugging Face.
- Startup, readiness y liveness probes.
- Solicitudes y límites de CPU y memoria.
- Ejecución como usuario no root `10001:10001`.

La imagen desplegada es:

```text
ghcr.io/maycol-rodriguez/dat-ia:0.2.0
```

## Requisitos

- Docker Desktop en modo Linux containers.
- `kubectl`.
- `kind`.
- Archivo local `.env` con las credenciales requeridas.
- Acceso a la imagen publicada en GitHub Container Registry.

El archivo `.env` no debe almacenarse en Git.

## Crear el clúster local

```powershell
kind create cluster `
    --name dat-ia `
    --wait 5m
```

Verificar el clúster:

```powershell
kubectl cluster-info `
    --context kind-dat-ia

kubectl get nodes
```

## Cargar la imagen en kind

```powershell
$image = "ghcr.io/maycol-rodriguez/dat-ia:0.2.0"

docker pull $image

kind load docker-image `
    $image `
    --name dat-ia
```

## Crear el Secret

El Secret se genera directamente desde el archivo local `.env`. Sus valores
no forman parte de los manifiestos versionados.

Primero se crea el namespace:

```powershell
kubectl apply `
    -f k8s/base/namespace.yaml
```

Después se crea o actualiza el Secret:

```powershell
kubectl create secret generic dat-ia-secrets `
    --namespace dat-ia `
    --from-env-file=".env" `
    --dry-run=client `
    -o yaml |
kubectl apply -f -
```

## Aplicar el despliegue

```powershell
kubectl apply `
    -k k8s/base
```

Esperar a que la API quede disponible:

```powershell
kubectl rollout status `
    deployment/dat-ia-api `
    --namespace dat-ia `
    --timeout=10m
```

Consultar los recursos:

```powershell
kubectl get `
    deployment,pod,service,pvc `
    --namespace dat-ia `
    -o wide
```

## Acceder a la API

El Service es de tipo `ClusterIP`. Para acceder desde la computadora local se
utiliza `port-forward`:

```powershell
kubectl port-forward `
    service/dat-ia-api `
    8000:80 `
    --namespace dat-ia
```

Direcciones disponibles:

- Health: `http://127.0.0.1:8000/health`
- Readiness: `http://127.0.0.1:8000/ready`
- Swagger: `http://127.0.0.1:8000/docs`
- Interfaz web: `http://127.0.0.1:8000/ui/`

## Prueba funcional

```powershell
$body = @{
    question = "¿Cuántas órdenes se registraron en total?"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/query/answer" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body
```

Durante la validación, Dat-IA generó y ejecutó:

```sql
SELECT COUNT(order_id) AS order_count
FROM olist_orders_dataset
```

El resultado fue de **99 441 órdenes**, coincidente con el Golden Set.

## Validaciones realizadas

El despliegue fue comprobado mediante las siguientes evidencias:

- El Deployment completó el rollout correctamente.
- El Pod quedó en estado `1/1 Running`.
- El contenedor se ejecutó sin privilegios como `uid=10001` y `gid=10001`.
- El endpoint `/health` respondió con estado `ok` y versión `0.2.0`.
- El endpoint `/ready` informó conexión activa con PostgreSQL.
- La consulta integral `/query/answer` devolvió estado `success`.
- ChromaDB indexó las 16 tablas del catálogo.
- SQLPromptShield se cargó correctamente.
- El PVC `dat-ia-chroma` quedó en estado `Bound`.
- La interfaz web `/ui/` quedó disponible mediante `port-forward`.

## Prueba de autorreparación y persistencia

Se eliminó manualmente el Pod de la API:

```powershell
kubectl delete pod `
    -l app.kubernetes.io/name=dat-ia-api `
    --namespace dat-ia
```

El Deployment creó automáticamente un nuevo Pod:

```powershell
kubectl wait `
    --for=condition=Ready `
    pod `
    --selector app.kubernetes.io/name=dat-ia-api `
    --namespace dat-ia `
    --timeout=10m
```

El nuevo Pod recuperó desde el PVC:

- los 16 esquemas almacenados en ChromaDB;
- una consulta registrada en Query Memory V2.

Los logs del segundo arranque mostraron:

```text
[startup] ChromaDB: 16 esquemas registrados.
[startup] Query memory V2: 1 consultas registradas.
[startup] SQLDatabase conectado (dialecto: postgresql).
INFO:     Application startup complete.
```

Esto comprobó la autorreparación del Deployment y la persistencia del estado
ante el reemplazo del contenedor.

## Validar los manifiestos

Validación local:

```powershell
kubectl apply `
    --dry-run=client `
    -f k8s/base/namespace.yaml

kubectl apply `
    --dry-run=client `
    -k k8s/base
```

Validación contra el servidor:

```powershell
kubectl apply `
    --dry-run=server `
    -f k8s/base/namespace.yaml

kubectl apply `
    --dry-run=server `
    -k k8s/base
```

Comprobar diferencias entre los manifiestos locales y el clúster:

```powershell
kubectl diff `
    -k k8s/base
```

## Revisar logs

```powershell
kubectl logs `
    deployment/dat-ia-api `
    --namespace dat-ia `
    --tail=150
```

## Eliminar el despliegue

```powershell
kubectl delete `
    -k k8s/base

kubectl delete secret `
    dat-ia-secrets `
    --namespace dat-ia

kubectl delete namespace dat-ia
```

Para eliminar también el clúster local:

```powershell
kind delete cluster `
    --name dat-ia
```

## Consideraciones

- El despliegue actual es local y fue validado sobre kind.
- El Service es interno (`ClusterIP`) y se accede mediante `kubectl port-forward`.
- Los secretos no se almacenan en Git.
- LangSmith permanece desactivado en esta configuración.
- La advertencia de Hugging Face por solicitudes sin autenticación no impidió
  la descarga ni la carga de SQLPromptShield.