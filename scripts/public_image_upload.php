<?php
declare(strict_types=1);

// Simple public image upload endpoint for ArcReel / Manxue video references.
// Deploy this file as: http://82.156.214.180:8887/video-image-upload.php

$UPLOAD_TOKEN = ''; // Optional: set a shared secret and configure MANXUE_PUBLIC_IMAGE_UPLOAD_TOKEN in ArcReel.
$PUBLIC_BASE_URL = 'http://82.156.214.180:8887';
$UPLOAD_DIR = __DIR__ . '/uploads';
$MAX_BYTES = 15 * 1024 * 1024;
$ALLOWED_MIME = [
    'image/jpeg' => 'jpg',
    'image/png' => 'png',
    'image/webp' => 'webp',
    'image/gif' => 'gif',
];

header('Content-Type: application/json; charset=utf-8');

function respond(int $status, array $body): never {
    http_response_code($status);
    echo json_encode($body, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    respond(405, ['ok' => false, 'error' => 'method_not_allowed']);
}

if ($UPLOAD_TOKEN !== '') {
    $headerToken = $_SERVER['HTTP_X_UPLOAD_TOKEN'] ?? '';
    $formToken = $_POST['token'] ?? '';
    if (!hash_equals($UPLOAD_TOKEN, $headerToken ?: $formToken)) {
        respond(401, ['ok' => false, 'error' => 'unauthorized']);
    }
}

if (!isset($_FILES['file']) || !is_uploaded_file($_FILES['file']['tmp_name'])) {
    respond(400, ['ok' => false, 'error' => 'missing_file']);
}

$file = $_FILES['file'];
if (($file['error'] ?? UPLOAD_ERR_OK) !== UPLOAD_ERR_OK) {
    respond(400, ['ok' => false, 'error' => 'upload_error', 'code' => $file['error']]);
}

$size = (int)($file['size'] ?? 0);
if ($size <= 0 || $size > $MAX_BYTES) {
    respond(413, ['ok' => false, 'error' => 'file_too_large', 'max_bytes' => $MAX_BYTES]);
}

$imageInfo = @getimagesize($file['tmp_name']);
$mime = is_array($imageInfo) && isset($imageInfo['mime']) ? (string)$imageInfo['mime'] : '';
if (!isset($ALLOWED_MIME[$mime])) {
    respond(415, ['ok' => false, 'error' => 'unsupported_media_type', 'mime' => $mime]);
}

if (!is_dir($UPLOAD_DIR) && !mkdir($UPLOAD_DIR, 0755, true)) {
    respond(500, ['ok' => false, 'error' => 'cannot_create_upload_dir']);
}

$ext = $ALLOWED_MIME[$mime];
$name = date('Ymd_His') . '_' . bin2hex(random_bytes(8)) . '.' . $ext;
$target = $UPLOAD_DIR . '/' . $name;

if (!move_uploaded_file($file['tmp_name'], $target)) {
    respond(500, ['ok' => false, 'error' => 'cannot_save_file']);
}

@chmod($target, 0644);

$url = rtrim($PUBLIC_BASE_URL, '/') . '/uploads/' . rawurlencode($name);
respond(200, [
    'ok' => true,
    'url' => $url,
    'mime' => $mime,
    'size' => $size,
]);
