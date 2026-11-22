extends Node
class_name TrainingLogReplayFetcher

## Fetch Godot ReplayRecord JSON files and RL turn traces directly from a local/VPN TrainingLog server.

@export var server_url: String = "http://127.0.0.1:8765"
@export var api_token: String = ""

signal replays_listed(replays: Array)
signal replay_loaded(replay_id: String, record: Variant, raw_bytes: PackedByteArray)
signal trace_loaded(replay_id: String, trace_payload: Dictionary)
signal request_failed(status_code: int, message: String)


func _headers() -> PackedStringArray:
	var hdrs := PackedStringArray(["Accept: application/json"])
	if api_token.strip_edges() != "":
		hdrs.append("Authorization: Bearer " + api_token.strip_edges())
	return hdrs


func list_replays(project: String = "", limit: int = 50, has_trace_only: bool = false) -> void:
	var params: PackedStringArray = ["limit=" + str(limit)]
	if project != "":
		params.append("project=" + project.uri_encode())
	if has_trace_only:
		params.append("has_trace=true")
	var url := server_url.rstrip("/") + "/api/replays?" + "&".join(params)
	_start_request(url, func(body: PackedByteArray):
		var parsed = JSON.parse_string(body.get_string_from_utf8())
		if typeof(parsed) == TYPE_DICTIONARY and parsed.has("replays"):
			replays_listed.emit(parsed["replays"])
		else:
			request_failed.emit(200, "Unexpected /api/replays payload")
	)


func fetch_replay(replay_id: String) -> void:
	var url := server_url.rstrip("/") + "/api/replays/" + replay_id.uri_encode() + "/file?inline=1"
	_start_request(url, func(body: PackedByteArray):
		var parsed = JSON.parse_string(body.get_string_from_utf8())
		replay_loaded.emit(replay_id, parsed, body)
	)


func fetch_trace(replay_id: String) -> void:
	var url := server_url.rstrip("/") + "/api/replays/" + replay_id.uri_encode() + "/trace"
	_start_request(url, func(body: PackedByteArray):
		var parsed = JSON.parse_string(body.get_string_from_utf8())
		if typeof(parsed) == TYPE_DICTIONARY:
			trace_loaded.emit(replay_id, parsed)
		else:
			request_failed.emit(200, "Unexpected trace JSON payload")
	)


func _start_request(url: String, on_ok: Callable) -> void:
	var http := HTTPRequest.new()
	add_child(http)
	http.request_completed.connect(
		func(result: int, code: int, _resp_headers: PackedStringArray, body: PackedByteArray):
			http.queue_free()
			if result != HTTPRequest.RESULT_SUCCESS or code < 200 or code >= 300:
				request_failed.emit(code, "Request failed (result=%d, http=%d): %s" % [result, code, url])
				return
			on_ok.call(body)
	)
	var err := http.request(url, _headers(), HTTPClient.METHOD_GET)
	if err != OK:
		http.queue_free()
		request_failed.emit(0, "Failed starting HTTPRequest: err=%d" % err)
